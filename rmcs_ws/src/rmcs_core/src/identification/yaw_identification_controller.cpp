#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <fast_tf/fast_tf.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_msgs/switch.hpp>
#include <rmcs_utility/csv_writer.hpp>

namespace rmcs_core::controller::identification {

namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
T parameter_or_declare(rclcpp::Node& node, const std::string& name, const T& default_value) {
    if (!node.has_parameter(name))
        node.declare_parameter<T>(name, default_value);
    return node.get_parameter(name).get_value<T>();
}

std::string timestamped_filename(const std::string& prefix) {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::ostringstream ss;
    // 带毫秒：避免同一秒内重复触发同一工况时覆盖已有 CSV
    ss << prefix << "_" << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S") << "-"
       << std::setw(3) << std::setfill('0') << milliseconds << ".csv";
    return ss.str();
}

double smootherstep(double u) {
    u = std::clamp(u, 0.0, 1.0);
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0);
}

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

/// 工况类型。序号同时写入 CSV 的 phase 列，沿用历史语义：
///   0 static（静置，零偏）  1 sine（正弦）  2 step（正负交替阶跃）  3 track（闭环跟踪 A/B）
///   4 rc（遥控随机控制：不施加脚本激励，yaw 控制权交还云台控制器，仅记录响应）
enum class ConditionType : int { kStatic = 0, kSine = 1, kStep = 2, kTrack = 3, kRc = 4 };

const char* condition_type_name(ConditionType type) {
    switch (type) {
    case ConditionType::kStatic: return "static";
    case ConditionType::kSine: return "sine";
    case ConditionType::kStep: return "step";
    case ConditionType::kTrack: return "track";
    case ConditionType::kRc: return "rc";
    }
    return "unknown";
}

struct Condition {
    int id = 0;
    ConditionType type = ConditionType::kStatic;
    double amplitude = 0.0;   // N*m，sine / step
    double frequency = 0.0;   // Hz，sine
    double duration = 0.0;    // s，全部类型
    double step_hold = 0.0;   // s，step 每拍时长
    int step_count = 0;       // step 拍数
    std::string control_mode; // baseline(无前馈) | model(带模型前馈)
    double kp_a = 0.0;        // track: 角度环增益覆盖（0=用 yaw_kp_angle_） // track: "baseline" | "model"
};

} // namespace

/// yaw 辨识与 A/B 对照协议控制器。
///
/// 两种工作模式：
///
/// 【1】传统模式（`condition_schedule` 为空）——行为与历史版本完全一致：
///   一次触发跑完整条 4 段协议，写单个 CSV：
///     phase 0 静置 static_duration_s
///     phase 1 正弦 sine_duration_s（幅值 sine_amplitude，频率 sine_frequency_hz）
///     phase 2 阶跃 step_count 拍 × step_duration_s（幅值 step_amplitude，正负交替）
///     phase 3 跟踪 track_duration_s（±track_step_angle_deg + track_sine_*）
///
/// 【2】扫描模式（`condition_schedule` 非空）——一次触发只跑列表里的**下一条**工况，
///   每条工况写**独立的 CSV**，跑完自动关闭；列表耗尽后不再启动。
///   触发手势不变：左拨杆 MIDDLE + 右拨杆 MIDDLE -> UP 的上升沿。
///   每次触发后必须把右拨杆拨回 MIDDLE，否则下一次不会触发。
///
/// `condition_schedule` 语法：一行一条工况，`#` 之后为注释，字段 `key=value`：
///     static dur=15
///     sine   amp=1.0 freq=0.20 dur=20
///     step   amp=2.0 hold=0.5 count=10        （dur 可省略，自动 = hold*count）
///     track  mode=baseline dur=20
///     rc     dur=60                            （遥控随机控制段，只记录不激励）
///   必填：static/sine/track/rc 需要 dur；sine 还需 amp+freq；step 需要 amp+hold+count；track 还需 mode。
///   可选：id=（默认按行号 1..N）。
///   任何语法/取值错误都会在**组件启动时**抛异常（快速失败，避免上车才发现）。
///
/// 安全：激励力矩硬限幅 torque_limit（track 段用 track_torque_limit），
/// |IMU 角速度| > abort_velocity 或遥控失效（含双下）自动中止并保留已写数据。
class YawIdentificationController
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    YawIdentificationController()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , static_duration_s_(parameter_or_declare(*this, "static_duration_s", 15.0))
        , sine_duration_s_(parameter_or_declare(*this, "sine_duration_s", 30.0))
        , sine_amplitude_(parameter_or_declare(*this, "sine_amplitude", 1.0))
        , sine_frequency_hz_(parameter_or_declare(*this, "sine_frequency_hz", 1.0))
        , step_amplitude_(parameter_or_declare(*this, "step_amplitude", 0.8))
        , step_duration_s_(parameter_or_declare(*this, "step_duration_s", 0.5))
        , step_count_(parameter_or_declare(*this, "step_count", 10))
        , torque_limit_(parameter_or_declare(*this, "torque_limit", 1.5))
        , abort_velocity_(parameter_or_declare(*this, "abort_velocity", 6.0))
        , csv_directory_(parameter_or_declare(*this, "csv_directory", std::string{"/tmp"}))
        , auto_advance_(parameter_or_declare(*this, "auto_advance", false))
        , inter_condition_gap_s_(parameter_or_declare(*this, "inter_condition_gap_s", 10.0))
        , abort_temperature_c_(parameter_or_declare(*this, "abort_temperature_c", 75.0))
        , yaw_excursion_limit_rad_(
              parameter_or_declare(*this, "yaw_excursion_limit_rad", 2.5))
        , start_from_condition_(parameter_or_declare(*this, "start_from_condition", 1))
        , track_duration_s_(parameter_or_declare(*this, "track_duration_s", 20.0))
        , track_step_angle_rad_(
              parameter_or_declare(*this, "track_step_angle_deg", 30.0) * std::numbers::pi_v<double> / 180.0)
        , track_step_hold_s_(parameter_or_declare(*this, "track_step_hold_s", 2.5))
        , track_sine_amplitude_rad_(
              parameter_or_declare(*this, "track_sine_amplitude_deg", 30.0) * std::numbers::pi_v<double> / 180.0)
        , track_sine_frequency_hz_(parameter_or_declare(*this, "track_sine_frequency_hz", 0.5))
        , track_transition_s_(parameter_or_declare(*this, "track_transition_s", 0.3))
        , track_torque_limit_(parameter_or_declare(*this, "track_torque_limit", 4.0))
        , control_mode_(parameter_or_declare(*this, "control_mode", std::string{"baseline"}))
        , model_J_(parameter_or_declare(*this, "model_J", 0.1503))
        , model_B_(parameter_or_declare(*this, "model_B", 0.6605))
        , model_tau_c_(parameter_or_declare(*this, "model_tau_c", 0.201))
        , yaw_kp_angle_(parameter_or_declare(*this, "yaw_kp_angle", 10.0))
        , yaw_kp_velocity_(parameter_or_declare(*this, "yaw_kp_velocity", 13.0))
        , yaw_ki_velocity_(parameter_or_declare(*this, "yaw_ki_velocity", 0.02))
        , yaw_velocity_integral_limit_(
              parameter_or_declare(*this, "yaw_velocity_integral_limit", 5.0)) {
        if (static_duration_s_ <= 0.0 || sine_duration_s_ <= 0.0 || step_duration_s_ <= 0.0
            || track_duration_s_ <= 0.0 || track_step_hold_s_ <= 0.0 || track_transition_s_ <= 0.0)
            throw std::runtime_error("durations must be positive");
        if (!(inter_condition_gap_s_ > 0.0))
            throw std::runtime_error("inter_condition_gap_s must be > 0");
        if (sine_frequency_hz_ <= 0.0 || track_sine_frequency_hz_ <= 0.0)
            throw std::runtime_error("frequencies must be positive");
        if (step_count_ < 1)
            throw std::runtime_error("step_count must be >= 1");
        if (!std::isfinite(sine_amplitude_) || !std::isfinite(step_amplitude_)
            || std::min(sine_amplitude_, step_amplitude_) < 0.0)
            throw std::runtime_error("amplitudes must be finite and non-negative");
        if (!std::isfinite(torque_limit_)
            || torque_limit_ < std::max(sine_amplitude_, step_amplitude_))
            throw std::runtime_error("torque_limit must be >= excitation amplitudes");
        if (control_mode_ != "baseline" && control_mode_ != "model")
            throw std::runtime_error("control_mode must be 'baseline' or 'model'");
        if (track_torque_limit_ < torque_limit_)
            throw std::runtime_error("track_torque_limit must be >= torque_limit");

        const auto schedule_text = parameter_or_declare(*this, "condition_schedule", std::string{});
        if (trim(schedule_text).empty()) {
            legacy_mode_ = true;
            schedule_ = legacy_schedule();
        } else {
            legacy_mode_ = false;
            schedule_ = parse_schedule(schedule_text);
        }
        validate_schedule();

        register_input("/predefined/timestamp", timestamp_);
        register_input("/remote/switch/left", switch_left_);
        register_input("/remote/switch/right", switch_right_);

        register_input("/gimbal/yaw/velocity_imu", yaw_velocity_imu_);
        register_input("/gimbal/yaw/velocity", yaw_velocity_);
        register_input("/gimbal/yaw/torque", yaw_torque_);
        register_input("/gimbal/yaw/angle", yaw_angle_);
        register_input("/gimbal/yaw/temperature", yaw_temperature_);
        register_input("/gimbal/pitch/angle", pitch_angle_);
        // pitch 监控：全部取硬件侧信号（读它们不构成循环依赖，云台控制器不依赖本组件以外的这些源）
        //   pitch_torque      = 电调反馈电流换算的实测力矩（保持炮管所需的重力力矩）
        //   pitch_temperature = 电调上报温度（用于排查过热降额）
        //   /tf               = 解算炮管世界俯仰角，便于与 pitch_gravity_ff_* 前馈项直接比对
        register_input("/gimbal/pitch/torque", pitch_torque_);
        register_input("/gimbal/pitch/temperature", pitch_temperature_);
        register_input("/tf", tf_);
        // 注意：本组件【不能】读 /gimbal/yaw/control_torque。
        // 云台控制器需要本组件的 identification_active / identification_torque 才能输出激励力矩，
        // 本组件若再读它的输出就构成循环依赖，执行器会在配对阶段直接卡死（一次 tick 都不跑）。
        // 激励期间 yaw 指令力矩恒等于 excitation_torque（交接逻辑直接转发），CSV 已记录该列。

        register_output("/gimbal/yaw/identification_torque", identification_torque_, nan_);
        register_output("/gimbal/yaw/identification_active", identification_active_, false);

        if (legacy_mode_) {
            RCLCPP_INFO(
                get_logger(),
                "Yaw identification: legacy mode (single trigger runs the full %zu-condition "
                "protocol), log dir=%s",
                schedule_.size(), csv_directory_.c_str());
        } else {
            RCLCPP_INFO(
                get_logger(),
                "Yaw identification: sweep mode, %zu conditions queued, log dir=%s "
                "(gesture: left=MIDDLE, right MIDDLE->UP; return right to MIDDLE after each run)",
                schedule_.size(), csv_directory_.c_str());
        }
        for (std::size_t i = 0; i < schedule_.size(); ++i) {
            const auto& condition = schedule_[i];
            RCLCPP_INFO(
                get_logger(),
                "  [%2zu/%zu] id=%d %-6s amp=%.2f freq=%.3f dur=%.1f hold=%.2f count=%d mode=%s",
                i + 1, schedule_.size(), condition.id, condition_type_name(condition.type),
                condition.amplitude, condition.frequency, condition.duration, condition.step_hold,
                condition.step_count,
                condition.control_mode.empty() ? "-" : condition.control_mode.c_str());
        }
    }

    ~YawIdentificationController() override { finish_test("destructor"); }

    void before_updating() override {
        finish_test("restart");
        *identification_torque_ = nan_;
        *identification_active_ = false;
        last_switch_left_ = rmcs_msgs::Switch::UNKNOWN;
        last_switch_right_ = rmcs_msgs::Switch::UNKNOWN;
        next_condition_ = start_from_condition_ <= 1
                            ? 0
                            : std::min<std::size_t>(
                                  static_cast<std::size_t>(start_from_condition_ - 1),
                                  schedule_.size() - 1);
        if (next_condition_ > 0)
            RCLCPP_INFO(
                get_logger(),
                "Yaw identification: start_from_condition=%d -> 下次触发从工况 id=%d (%s) 开始",
                start_from_condition_, schedule_[next_condition_].id,
                condition_type_name(schedule_[next_condition_].type));
    }

    void update() override {
        const auto current_switch_left = *switch_left_;
        const auto current_switch_right = *switch_right_;

        const bool remote_enabled =
            !(current_switch_left == rmcs_msgs::Switch::UNKNOWN
              || current_switch_right == rmcs_msgs::Switch::UNKNOWN
              || (current_switch_left == rmcs_msgs::Switch::DOWN
                  && current_switch_right == rmcs_msgs::Switch::DOWN));

        if (!remote_enabled) {
            abort_test("remote disabled");
            *identification_torque_ = nan_;
            *identification_active_ = false;
            store_switch_state(current_switch_left, current_switch_right);
            return;
        }

        // 温度保护：运行/间隔期间任一电机温度达到上限，立即中止整场测试（已采集的 CSV 保留）
        if (abort_temperature_c_ > 0.0 && (phase_ == Phase::kRunning || phase_ == Phase::kGap)) {
            const double yaw_t = *yaw_temperature_;
            const double pitch_t = *pitch_temperature_;
            const double hottest = std::max(yaw_t, pitch_t);
            if (std::isfinite(hottest) && hottest >= abort_temperature_c_) {
                RCLCPP_ERROR(
                    get_logger(),
                    "Yaw identification: ABORT — motor temperature %.0f C reached the %.0f C limit "
                    "(yaw %.0f C, pitch %.0f C). Let the motors cool down before resuming.",
                    hottest, abort_temperature_c_, yaw_t, pitch_t);
                finish_test("temperature limit");
                *identification_torque_ = nan_;
                *identification_active_ = false;
                store_switch_state(current_switch_left, current_switch_right);
                return;
            }
        }

        if (phase_ == Phase::kIdle) {
            if (should_start_test(current_switch_left, current_switch_right)) {
                if (legacy_mode_)
                    next_condition_ = 0; // 传统模式：一次触发跑完 4 段
                if (next_condition_ >= schedule_.size()) {
                    // 扫描跑完后再拨一次 = 从第 1 条重来（用于补采 / 加热循环），CSV 各自独立不会覆盖
                    RCLCPP_WARN(
                        get_logger(),
                        "Yaw identification: sweep already finished (%zu/%zu); restarting from "
                        "condition 1.",
                        next_condition_, schedule_.size());
                    next_condition_ = 0;
                }
                start_test(next_condition_);
            }
            if (phase_ == Phase::kIdle) {
                *identification_torque_ = nan_;
                *identification_active_ = false;
                store_switch_state(current_switch_left, current_switch_right);
                return;
            }
        }

        if (phase_ == Phase::kGap) {
            // 静置间隔：yaw 交还遥控、停止记录，等系统彻底静止后再自动开始下一条工况
            *identification_torque_ = nan_;
            *identification_active_ = false;
            const double gap_elapsed_s =
                std::chrono::duration<double>(*timestamp_ - gap_start_time_).count();
            if (gap_elapsed_s >= inter_condition_gap_s_ && next_condition_ < schedule_.size()) {
                RCLCPP_INFO(
                    get_logger(),
                    "Yaw identification: settling gap done (%.1f s), starting condition id=%d (%s)",
                    gap_elapsed_s, schedule_[next_condition_].id,
                    condition_type_name(schedule_[next_condition_].type));
                start_test(next_condition_);
            }
            if (phase_ == Phase::kGap) {
                store_switch_state(current_switch_left, current_switch_right);
                return;
            }
        }

        const double elapsed_s =
            std::chrono::duration<double>(*timestamp_ - test_start_time_).count();

        update_continuous_angle();

        const Condition& condition = schedule_[current_condition_];
        const double local_s = elapsed_s - condition_start_elapsed_s_;

        // 记录本条工况的 yaw 起始角，用于限制摆动行程。
        // 低频大扭矩工况会让 yaw 朝一个方向持续转出很大角度（例如 0.05 Hz @ 1 N·m
        // 约 0.8 圈），实测会把线缆拉紧 / 产生机械干涉——先把 pitch 拽到机械下限，
        // 再把 yaw 一起卡死。因此按工况限制行程，超限立即中止并提示检查。
        if (condition_origin_id_ != current_condition_) {
            condition_origin_id_ = current_condition_;
            condition_origin_angle_ = continuous_angle_;
        }

        double excitation = 0.0;
        double ff_torque = nan_;
        double ref_angle = nan_;
        double ref_velocity = nan_;
        double ref_acceleration = nan_;
        bool condition_finished = false;

        switch (condition.type) {
        case ConditionType::kStatic:
            if (local_s >= condition.duration)
                condition_finished = true;
            break;

        case ConditionType::kSine:
            if (local_s >= condition.duration) {
                condition_finished = true;
            } else {
                excitation = condition.amplitude
                           * std::sin(
                               2.0 * std::numbers::pi_v<double> * condition.frequency * local_s);
            }
            break;

        case ConditionType::kStep: {
            const auto step_index = static_cast<int>(local_s / condition.step_hold);
            if (step_index >= condition.step_count) {
                condition_finished = true;
            } else {
                excitation = (step_index % 2 == 0) ? condition.amplitude : -condition.amplitude;
            }
            break;
        }

        case ConditionType::kTrack: {
            if (local_s >= condition.duration) {
                condition_finished = true;
                break;
            }
            if (!tracking_initialized_) {
                tracking_initialized_ = true;
                track_ref_origin_ = continuous_angle_;
                reference_angle_ = track_ref_origin_;
                reference_velocity_ = 0.0;
                previous_velocity_ref_ = 0.0;
                velocity_integrator_ = 0.0;
            }
            update_tracking_reference(local_s);
            const double dt = std::max(update_dt(), 1e-6);
            ref_acceleration = (reference_velocity_ - previous_velocity_ref_) / dt;
            previous_velocity_ref_ = reference_velocity_;

            const double angle_error = reference_angle_ - continuous_angle_;
            const double velocity_error = reference_velocity_ - *yaw_velocity_imu_;

            // 复刻原控制器：级联 PID，速度积分按原 PidCalculator 语义直接累加误差（不乘 dt）
            velocity_integrator_ = std::clamp(
                velocity_integrator_ + velocity_error, -yaw_velocity_integral_limit_,
                yaw_velocity_integral_limit_);
            const double kp_angle = condition.kp_a > 0.0 ? condition.kp_a : yaw_kp_angle_;
            double torque = yaw_kp_velocity_ * (kp_angle * angle_error + velocity_error)
                          + yaw_ki_velocity_ * velocity_integrator_;

            if (condition.control_mode == "model") {
                ff_torque = model_J_ * ref_acceleration + model_B_ * reference_velocity_
                          + model_tau_c_ * ((reference_velocity_ >= 0.0) ? 1.0 : -1.0);
                torque += ff_torque;
            }

            torque = std::clamp(torque, -track_torque_limit_, track_torque_limit_);
            excitation = torque;
            ref_angle = reference_angle_;
            ref_velocity = reference_velocity_;
            break;
        }

        case ConditionType::kRc:
            // 遥控随机控制段：不施加任何脚本激励，yaw 控制权交还云台控制器（遥控驱动），
            // 本组件只记录响应。实际作用力矩见 CSV 的 measured_torque 列（电调反馈电流换算）。
            if (local_s >= condition.duration)
                condition_finished = true;
            break;
        }

        if (condition_finished) {
            *identification_torque_ = nan_;
            *identification_active_ = false;
            if (current_condition_ + 1 < schedule_.size() && legacy_mode_) {
                // 传统模式：同一条 CSV 内推进到下一段
                current_condition_ += 1;
                condition_start_elapsed_s_ = elapsed_s;
                tracking_initialized_ = false;
                velocity_integrator_ = 0.0;
                previous_velocity_ref_ = 0.0;
            } else {
                if (!legacy_mode_)
                    next_condition_ = current_condition_ + 1;
                RCLCPP_INFO(
                    get_logger(), "Yaw identification: condition id=%d (%s) finished at %.2fs",
                    condition.id, condition_type_name(condition.type), elapsed_s);
                finish_test("condition finished");
                // 自动连续模式：插入静置间隔后自动开下一条；否则停下等下一次拨杆触发
                if (auto_advance_ && !legacy_mode_ && next_condition_ < schedule_.size()) {
                    phase_ = Phase::kGap;
                    gap_start_time_ = *timestamp_;
                    RCLCPP_INFO(
                        get_logger(),
                        "Yaw identification: %.1f s settling gap, then condition id=%d (%s) starts "
                        "automatically",
                        inter_condition_gap_s_, schedule_[next_condition_].id,
                        condition_type_name(schedule_[next_condition_].type));
                }
            }
            store_switch_state(current_switch_left, current_switch_right);
            return;
        }

        // 行程保护：单条工况内 yaw 偏转超限立即中止（线缆缠绕 / 机械干涉的安全网）
        {
            const double excursion_rad = std::abs(continuous_angle_ - condition_origin_angle_);
            if (yaw_excursion_limit_rad_ > 0.0 && excursion_rad > yaw_excursion_limit_rad_) {
                log_sample(
                    elapsed_s, local_s, condition, excitation, ff_torque, ref_angle, ref_velocity,
                    ref_acceleration);
                RCLCPP_ERROR(
                    get_logger(),
                    "Yaw identification: ABORT — yaw excursion %.1f deg exceeds the %.1f deg limit "
                    "(condition id=%d, %s). 疑似线缆缠绕或机械干涉，请检查后再继续。",
                    excursion_rad * 180.0 / std::numbers::pi_v<double>,
                    yaw_excursion_limit_rad_ * 180.0 / std::numbers::pi_v<double>, condition.id,
                    condition_type_name(condition.type));
                *identification_torque_ = nan_;
                *identification_active_ = false;
                abort_test("yaw excursion limit");
                store_switch_state(current_switch_left, current_switch_right);
                return;
            }
        }

        const double limit =
            (condition.type == ConditionType::kTrack) ? track_torque_limit_ : torque_limit_;
        if (condition.type == ConditionType::kRc) {
            // RC 段不接管 yaw：identification_active=false，云台控制器按遥控正常闭环
            *identification_torque_ = nan_;
            *identification_active_ = false;
        } else {
            excitation = std::clamp(excitation, -limit, limit);
            *identification_torque_ = excitation;
            *identification_active_ = true;
        }

        if (std::abs(*yaw_velocity_imu_) > abort_velocity_) {
            log_sample(
                elapsed_s, local_s, condition, excitation, ff_torque, ref_angle, ref_velocity,
                ref_acceleration);
            abort_test("imu velocity over limit");
            *identification_torque_ = nan_;
            *identification_active_ = false;
            store_switch_state(current_switch_left, current_switch_right);
            return;
        }

        log_sample(
            elapsed_s, local_s, condition, excitation, ff_torque, ref_angle, ref_velocity,
            ref_acceleration);

        store_switch_state(current_switch_left, current_switch_right);
    }

private:
    static constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();
    static constexpr auto kFlushInterval = std::chrono::duration<double>(0.1);

    enum class Phase { kIdle, kRunning, kGap };

    /// 传统模式的 4 段协议（与历史版本逐位一致）
    std::vector<Condition> legacy_schedule() const {
        std::vector<Condition> schedule;
        schedule.push_back(Condition{
            1, ConditionType::kStatic, 0.0, 0.0, static_duration_s_, 0.0, 0, {}});
        schedule.push_back(Condition{
            2, ConditionType::kSine, sine_amplitude_, sine_frequency_hz_, sine_duration_s_, 0.0, 0,
            {}});
        schedule.push_back(Condition{
            3, ConditionType::kStep, step_amplitude_, 0.0, step_duration_s_ * step_count_,
            step_duration_s_, step_count_, {}});
        schedule.push_back(Condition{
            4, ConditionType::kTrack, 0.0, 0.0, track_duration_s_, 0.0, 0, control_mode_});
        return schedule;
    }

    /// 解析 condition_schedule 文本；任何错误直接抛异常
    std::vector<Condition> parse_schedule(const std::string& text) const {
        std::vector<Condition> schedule;
        std::istringstream stream{text};
        std::string line;
        int line_number = 0;
        while (std::getline(stream, line)) {
            ++line_number;
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line = line.substr(0, comment);
            line = trim(line);
            if (line.empty())
                continue;

            std::istringstream tokens{line};
            std::string type_token;
            tokens >> type_token;

            Condition condition;
            condition.id = static_cast<int>(schedule.size()) + 1;
            if (type_token == "static")
                condition.type = ConditionType::kStatic;
            else if (type_token == "sine")
                condition.type = ConditionType::kSine;
            else if (type_token == "step")
                condition.type = ConditionType::kStep;
            else if (type_token == "track")
                condition.type = ConditionType::kTrack;
            else if (type_token == "rc")
                condition.type = ConditionType::kRc;
            else
                throw std::runtime_error(
                    "condition_schedule line " + std::to_string(line_number)
                    + ": unknown type '" + type_token
                    + "' (expected static|sine|step|track|rc)");

            std::string token;
            while (tokens >> token) {
                const auto equals = token.find('=');
                if (equals == std::string::npos)
                    throw std::runtime_error(
                        "condition_schedule line " + std::to_string(line_number)
                        + ": expected key=value, got '" + token + "'");
                const auto key = token.substr(0, equals);
                const auto value = token.substr(equals + 1);
                auto to_double = [&]() {
                    try {
                        std::size_t consumed = 0;
                        const double parsed = std::stod(value, &consumed);
                        if (consumed != value.size())
                            throw std::invalid_argument("trailing characters");
                        return parsed;
                    } catch (const std::exception&) {
                        throw std::runtime_error(
                            "condition_schedule line " + std::to_string(line_number) + ": bad value '"
                            + value + "' for key '" + key + "'");
                    }
                };

                if (key == "id")
                    condition.id = static_cast<int>(to_double());
                else if (key == "amp" || key == "amplitude")
                    condition.amplitude = to_double();
                else if (key == "freq" || key == "frequency_hz")
                    condition.frequency = to_double();
                else if (key == "dur" || key == "duration_s")
                    condition.duration = to_double();
                else if (key == "hold" || key == "step_duration_s")
                    condition.step_hold = to_double();
                else if (key == "count" || key == "step_count")
                    condition.step_count = static_cast<int>(to_double());
                else if (key == "mode" || key == "control_mode")
                    condition.control_mode = value;
                else if (key == "kp_a" || key == "angle_kp")
                    condition.kp_a = to_double();
                else
                    throw std::runtime_error(
                        "condition_schedule line " + std::to_string(line_number)
                        + ": unknown key '" + key + "'");
            }

            if (condition.type == ConditionType::kStep) {
                if (!(condition.amplitude > 0.0))
                    throw std::runtime_error(
                        "condition_schedule line " + std::to_string(line_number)
                        + ": step requires amp>0");
                if (!(condition.step_hold > 0.0))
                    throw std::runtime_error(
                        "condition_schedule line " + std::to_string(line_number)
                        + ": step requires hold>0");
                if (condition.step_count < 1)
                    throw std::runtime_error(
                        "condition_schedule line " + std::to_string(line_number)
                        + ": step requires count>=1");
                // step 的 dur 可省略：由 hold * count 推出（避免两处写法互相矛盾）
                if (!(condition.duration > 0.0))
                    condition.duration =
                        condition.step_hold * static_cast<double>(condition.step_count);
            } else if (!(condition.duration > 0.0)) {
                throw std::runtime_error(
                    "condition_schedule line " + std::to_string(line_number)
                    + ": 'dur' must be > 0");
            }
            if (condition.type == ConditionType::kSine
                && (!(condition.amplitude > 0.0) || !(condition.frequency > 0.0)))
                throw std::runtime_error(
                    "condition_schedule line " + std::to_string(line_number)
                    + ": sine requires amp>0 and freq>0");
            if (condition.type == ConditionType::kTrack
                && condition.control_mode != "baseline" && condition.control_mode != "model")
                throw std::runtime_error(
                    "condition_schedule line " + std::to_string(line_number)
                    + ": track requires mode=baseline|model");

            schedule.push_back(condition);
        }

        if (schedule.empty())
            throw std::runtime_error("condition_schedule is not empty but contains no conditions");
        return schedule;
    }

    /// 逐条校验幅值不超过硬限幅——宁可启动就报错，也不要上车后静默削顶
    void validate_schedule() const {
        for (const auto& condition : schedule_) {
            if (condition.type == ConditionType::kStatic)
                continue;
            const double limit =
                (condition.type == ConditionType::kTrack) ? track_torque_limit_ : torque_limit_;
            if (condition.amplitude > limit)
                throw std::runtime_error(
                    "condition id=" + std::to_string(condition.id) + " ("
                    + condition_type_name(condition.type) + ") amplitude "
                    + std::to_string(condition.amplitude)
                    + " exceeds the configured limit " + std::to_string(limit)
                    + " — raise torque_limit/track_torque_limit or lower the amplitude");
        }
    }

    /// 炮管方向在世界系（OdomImu）中的俯仰角——与云台控制器重力前馈
    /// pitch_gravity_ff_gain * sin(world_pitch - pitch_gravity_ff_phase) 用的是同一个量。
    /// 用它可以判定前馈项是否恰好等于保持炮管所需的实测重力力矩。
    double pitch_world_angle() const {
        auto dir = fast_tf::cast<rmcs_description::OdomImu>(
            rmcs_description::PitchLink::DirectionVector{Eigen::Vector3d::UnitX()}, *tf_);
        return std::asin(std::clamp(dir->z(), -1.0, 1.0));
    }

    void update_continuous_angle() {
        const double raw = *yaw_angle_;
        if (!angle_initialized_) {
            continuous_angle_ = raw;
            previous_raw_angle_ = raw;
            angle_initialized_ = true;
            return;
        }
        continuous_angle_ += std::remainder(raw - previous_raw_angle_, 2.0 * std::numbers::pi_v<double>);
        previous_raw_angle_ = raw;
    }

    void update_tracking_reference(double local_s) {
        const double step_phase_s = 4.0 * track_step_hold_s_;
        constexpr double kTargets[4] = {1.0, -1.0, 1.0, 0.0};

        if (local_s < step_phase_s) {
            const auto seg = static_cast<int>(local_s / track_step_hold_s_);
            const double seg_start = static_cast<double>(seg) * track_step_hold_s_;
            const double u = smootherstep((local_s - seg_start) / track_transition_s_);
            const double prev = (seg == 0) ? 0.0 : kTargets[seg - 1];
            reference_offset_ =
                (prev + (kTargets[seg] - prev) * u) * track_step_angle_rad_;
        } else {
            const double local_sine = local_s - step_phase_s;
            reference_offset_ = track_sine_amplitude_rad_
                              * std::sin(2.0 * std::numbers::pi_v<double> * track_sine_frequency_hz_
                                         * local_sine);
        }

        const double new_ref = track_ref_origin_ + reference_offset_;
        const double dt = std::max(update_dt(), 1e-6);
        reference_velocity_ = (new_ref - reference_angle_) / dt;
        reference_angle_ = new_ref;
    }

    bool should_start_test(
        rmcs_msgs::Switch current_switch_left, rmcs_msgs::Switch current_switch_right) const {
        using rmcs_msgs::Switch;
        return last_switch_left_ == Switch::MIDDLE && last_switch_right_ == Switch::MIDDLE
            && current_switch_left == Switch::MIDDLE && current_switch_right == Switch::UP;
    }

    void start_test(std::size_t condition_index) {
        current_condition_ = condition_index;
        const Condition& condition = schedule_[current_condition_];

        std::ostringstream prefix;
        prefix << "yaw";
        if (legacy_mode_) {
            prefix << "_identification";
        } else {
            prefix << "_c" << std::setw(2) << std::setfill('0') << condition.id << "_"
                   << condition_type_name(condition.type);
        }
        const auto path = std::filesystem::path{csv_directory_} / timestamped_filename(prefix.str());

        try {
            csv_writer_.open(path);
        } catch (const std::exception& exception) {
            RCLCPP_ERROR(
                get_logger(), "Failed to open identification log '%s': %s", path.string().c_str(),
                exception.what());
            return;
        }
        try {
            csv_writer_.write_row(
                "elapsed_s", "phase", "excitation_torque", "ff_torque", "reference_angle",
                "reference_velocity", "reference_acceleration", "measured_torque",
                "measured_velocity", "measured_velocity_imu", "measured_angle", "pitch_angle",
                "temperature", "condition_id", "condition_type", "condition_elapsed_s",
                "wallclock_s", "pitch_torque", "pitch_temperature", "pitch_world_angle");
            csv_writer_.flush();
        } catch (const std::exception& exception) {
            RCLCPP_ERROR(get_logger(), "Failed to write identification log header: %s", exception.what());
            csv_writer_.close();
            return;
        }

        test_start_time_ = *timestamp_;
        test_start_wallclock_s_ = std::chrono::duration<double>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
        next_flush_time_ =
            test_start_time_ + std::chrono::duration_cast<Clock::duration>(kFlushInterval);
        phase_ = Phase::kRunning;
        condition_start_elapsed_s_ = 0.0;
        tracking_initialized_ = false;
        angle_initialized_ = false;
        reference_angle_ = 0.0;
        reference_velocity_ = 0.0;
        previous_velocity_ref_ = 0.0;
        velocity_integrator_ = 0.0;
        *identification_active_ = true;
        RCLCPP_INFO(
            get_logger(), "Yaw identification: condition %zu/%zu started (id=%d %s mode=%s), log=%s",
            condition_index + 1, schedule_.size(), condition.id,
            condition_type_name(condition.type),
            condition.control_mode.empty() ? "-" : condition.control_mode.c_str(),
            path.string().c_str());
    }

    void log_sample(
        double elapsed_s, double local_s, const Condition& condition, double excitation,
        double ff_torque, double ref_angle, double ref_velocity, double ref_acceleration) {
        if (phase_ != Phase::kRunning || !csv_writer_.is_open())
            return;
        try {
            csv_writer_.write_row(
                elapsed_s, static_cast<int>(condition.type), excitation, ff_torque, ref_angle,
                ref_velocity, ref_acceleration, *yaw_torque_, *yaw_velocity_, *yaw_velocity_imu_,
                *yaw_angle_, *pitch_angle_, *yaw_temperature_, condition.id,
                condition_type_name(condition.type), local_s, test_start_wallclock_s_ + elapsed_s,
                *pitch_torque_, *pitch_temperature_, pitch_world_angle());
        } catch (const std::exception& exception) {
            RCLCPP_ERROR(get_logger(), "Failed to write identification sample: %s", exception.what());
        }
        if (*timestamp_ >= next_flush_time_) {
            csv_writer_.flush();
            while (*timestamp_ >= next_flush_time_)
                next_flush_time_ += std::chrono::duration_cast<Clock::duration>(kFlushInterval);
        }
    }

    void finish_test(const char* reason) {
        if (phase_ != Phase::kRunning)
            return;
        phase_ = Phase::kIdle;
        *identification_active_ = false;
        *identification_torque_ = nan_;
        if (csv_writer_.is_open()) {
            csv_writer_.flush();
            csv_writer_.close();
            RCLCPP_INFO(get_logger(), "Yaw identification test finished (%s), log=%s", reason,
                        csv_writer_.path().string().c_str());
        }
    }

    void abort_test(const char* reason) { finish_test(reason); }

    void store_switch_state(
        rmcs_msgs::Switch current_switch_left, rmcs_msgs::Switch current_switch_right) {
        last_switch_left_ = current_switch_left;
        last_switch_right_ = current_switch_right;
    }

    double update_dt() const { return 0.001; }

    const double static_duration_s_;
    const double sine_duration_s_;
    const double sine_amplitude_;
    const double sine_frequency_hz_;
    const double step_amplitude_;
    const double step_duration_s_;
    const int step_count_;
    const double torque_limit_;
    const double abort_velocity_;
    const std::string csv_directory_;
    const bool auto_advance_;              // 一次触发跑完全部工况（条目间插静置间隔）
    const double inter_condition_gap_s_;   // 相邻工况之间的静置间隔
    const double abort_temperature_c_;     // 电机温度上限（<=0 表示不启用）
    const double yaw_excursion_limit_rad_; // 单条工况内 yaw 允许的偏转行程（<=0 表示不启用）
    const int start_from_condition_;       // 从第几条工况开始（1 起；>1 用于中止后续跑）

    const double track_duration_s_;
    const double track_step_angle_rad_;
    const double track_step_hold_s_;
    const double track_sine_amplitude_rad_;
    const double track_sine_frequency_hz_;
    const double track_transition_s_;
    const double track_torque_limit_;
    const std::string control_mode_;
    const double model_J_;
    const double model_B_;
    const double model_tau_c_;
    const double yaw_kp_angle_;
    const double yaw_kp_velocity_;
    const double yaw_ki_velocity_;
    const double yaw_velocity_integral_limit_;

    std::vector<Condition> schedule_;
    bool legacy_mode_ = true;
    std::size_t next_condition_ = 0;
    std::size_t current_condition_ = 0;
    double condition_start_elapsed_s_ = 0.0;

    Phase phase_ = Phase::kIdle;
    Clock::time_point test_start_time_{};
    Clock::time_point next_flush_time_{};
    Clock::time_point gap_start_time_{};
    double test_start_wallclock_s_ = 0.0;

    bool angle_initialized_ = false;
    double continuous_angle_ = 0.0;
    double condition_origin_angle_ = 0.0; // 本条工况开始时的 yaw 连续角
    std::size_t condition_origin_id_ = std::numeric_limits<std::size_t>::max();
    double previous_raw_angle_ = 0.0;

    bool tracking_initialized_ = false;
    double track_ref_origin_ = 0.0;
    double reference_offset_ = 0.0;
    double reference_angle_ = 0.0;
    double reference_velocity_ = 0.0;
    double previous_velocity_ref_ = 0.0;
    double velocity_integrator_ = 0.0;

    rmcs_msgs::Switch last_switch_left_ = rmcs_msgs::Switch::UNKNOWN;
    rmcs_msgs::Switch last_switch_right_ = rmcs_msgs::Switch::UNKNOWN;

    InputInterface<Clock::time_point> timestamp_;
    InputInterface<rmcs_msgs::Switch> switch_left_;
    InputInterface<rmcs_msgs::Switch> switch_right_;

    InputInterface<double> yaw_velocity_imu_;
    InputInterface<double> yaw_velocity_;
    InputInterface<double> yaw_torque_;
    InputInterface<double> yaw_angle_;
    InputInterface<double> yaw_temperature_;
    InputInterface<double> pitch_angle_;
    InputInterface<double> pitch_torque_;
    InputInterface<double> pitch_temperature_;
    InputInterface<rmcs_description::Tf> tf_;

    OutputInterface<double> identification_torque_;
    OutputInterface<bool> identification_active_;

    rmcs_utility::CsvWriter csv_writer_;
};

} // namespace rmcs_core::controller::identification

PLUGINLIB_EXPORT_CLASS(
    rmcs_core::controller::identification::YawIdentificationController,
    rmcs_executor::Component)
