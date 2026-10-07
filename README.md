# 步兵 C 云台 Yaw 轴系统辨识与控制调优（RMCS `merge/deformable`）

本仓库为 RMCS 培训作业：对步兵 C 全向车（`deformable-infantry-omni-c`）云台 yaw 轴建立**二阶 + 库仑摩擦模型**，用实车数据完成系统辨识，并基于辨识结果进行控制器调优与 A/B 对照验证。

## 报告

- 网页版（GitHub 直接渲染）：[docs/zh-cn/c_watch.md](docs/zh-cn/c_watch.md)
- PDF 版：[步兵C_yaw系统辨识与控制优化报告.pdf](docs/zh-cn/步兵C_yaw系统辨识与控制优化报告.pdf)

## 核心结果

- **统一模型**（输出误差法，两轮数据联合辨识，跨数据集验证 + 自然遥控数据波形相关 0.871）：

  ```
  J·ω̇ + B·ω + τ_c·sign(ω) = τ_applied
      J   = 0.137 kg·m²          等效转动惯量
      B   = 0.530 N·m·s/rad      等效黏性阻尼
      τ_c = 0.360 N·m            库仑摩擦力矩
  ```

- **控制器调优**：由模型解析推导 kp_a 10→25（ζ=1）+ 模型前馈 `τ_ff = J·α_ref + B·ω_ref + τ_c·sign(ω_ref)`；三轮 A/B 对照——力矩余量充足时跟踪 RMS **−33.4%**、阶跃超调 **−77.7%**；限幅受限时 **+46% 负优化**。核心结论：**模型整定与前馈的收益取决于执行器余量，调参前先校验执行器预算**。
- **平台硬约束**：yaw 行程 143° 安全上限（~288° 线缆拉紧导致卡死）；pitch 力矩余量仅 ~1 N·m（根治需机械重力配平）。

## 主要文件

| 文件 | 说明 |
|---|---|
| `rmcs_ws/src/rmcs_bringup/config/deformable-infantry-omni-c-id.yaml` | 试验配置（下层解耦 + 28 工况协议 + 连续模式 + 四重安全保护） |
| `rmcs_ws/src/rmcs_core/src/identification/yaw_identification_controller.cpp` | 标准激励协议 / 扫描 / 连续 / rc 工况 / 保护组件 |
| `rmcs_ws/src/rmcs_core/src/debug/value_collector.cpp` | 数据采集（增加时间戳与 pitch 监控列） |
| `data/sessionA\|B\|C/` | 三轮实机采集数据（1000 Hz，含 pitch 监控列） |
| `.analyze_yaw.py` 等（仓库根目录） | 辨识分析、拟合与 A/B 对比脚本 |

## 复现

见报告附录 B：切换试验配置 → 触发（左拨杆中位 + 右拨杆中→上，一次触发自动跑完全部工况）→ `scp remote:/tmp/yaw_c*.csv` 取数 → 恢复比赛配置。
