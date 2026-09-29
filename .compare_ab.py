import csv
import math
import sys

import numpy as np

TRACK_START = 50.0
HOLD = 2.5
STEP_DEG = 30.0
SINE_START = 10.0


def load(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        names = {n: i for i, n in enumerate(header)}
        parsed = ([float(x) if x not in ("", "nan", "-nan") else math.nan for x in row]
                  for row in reader)
        rows = np.array([r + [math.nan] * (len(header) - len(r)) for r in parsed])
    g = lambda n: rows[:, names[n]]
    return {
        "t": g("elapsed_s"), "phase": g("phase"), "tau": g("excitation_torque"),
        "ff": g("ff_torque"), "ref": g("reference_angle"),
        "ref_v": g("reference_velocity"), "ref_a": g("reference_acceleration"),
        "tau_fb": g("measured_torque"), "w": g("measured_velocity_imu"),
        "ang": g("measured_angle"),
    }


def wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def tracking_metrics(data, label):
    m = (data["phase"] == 3) & ~np.isnan(data["ref"])
    t = data["t"][m] - TRACK_START
    ref = data["ref"][m]
    err = wrap(ref - data["ang"][m])
    tau = data["tau"][m]
    print(f"\n===== {label} =====")
    print(f"tracking rows: {m.sum()}, duration {t[-1] - t[0]:.1f}s")

    # 阶跃段 4 拍
    print("--- step segments ---")
    targets = [+STEP_DEG, -STEP_DEG, +STEP_DEG, 0.0]
    for k in range(4):
        s = k * HOLD
        seg = (t >= s) & (t < s + HOLD)
        if seg.sum() < 100:
            continue
        tt, ee, rr, tt_tau = t[seg], err[seg], ref[seg], tau[seg]
        target_rad = np.radians(targets[k])
        final_err = np.degrees(np.mean(ee[tt > s + HOLD - 0.5]))
        # 超调: 超过目标方向的最大过冲
        beyond = (ee * np.sign(target_rad) < 0) if abs(target_rad) > 0 else None
        overshoot = np.degrees(np.max(np.abs(ee[np.sign(ee) != np.sign(target_rad)]))) if beyond is not None and beyond.any() else 0.0
        # 上升时间: 从过渡开始到 |err| 首次 < 10% 目标
        tol = abs(target_rad) * 0.1 if target_rad != 0 else np.radians(2)
        reach = np.where(np.abs(ee) < tol)[0]
        rise = (tt[reach[0]] - s) * 1000 if len(reach) else float("nan")
        # 稳态时间: 最后一次 |err|>=2deg 的时刻
        out = np.where(np.abs(ee) >= np.radians(2))[0]
        settle = (tt[out[-1]] - s) * 1000 if len(out) else 0.0
        print(f"  seg{k} target={targets[k]:+.0f}deg: final_err={final_err:+.3f}deg  "
              f"rise(10%)={rise:.0f}ms  settle(2deg)={settle:.0f}ms  overshoot={overshoot:.2f}deg")

    # 正弦段
    s = SINE_START
    seg = t >= s
    if seg.sum() > 100:
        rms = np.degrees(np.sqrt(np.mean(err[seg] ** 2)))
        print(f"--- sine tracking ---")
        print(f"  RMS err={rms:.3f}deg  max|err|={np.degrees(np.max(np.abs(err[seg]))):.3f}deg")

    # 总体
    print("--- overall & torque ---")
    print(f"  tracking RMS err: {np.degrees(np.sqrt(np.nanmean(err ** 2))):.3f}deg  "
          f"max: {np.degrees(np.nanmax(np.abs(err))):.3f}deg")
    print(f"  tau: rms={np.sqrt(np.nanmean(tau ** 2)):.3f}  peak={np.nanmax(np.abs(tau)):.3f} N*m  "
          f"smoothness(std of diff)={np.nanstd(np.diff(tau)) * 1000:.2f} mN*m/ms")
    return err, t, tau


base = load("/workspaces/RMCS/data/yaw_baseline_2026-09-26.csv")
model = load("/workspaces/RMCS/data/yaw_model_2026-09-26.csv")

# 激励段一致性 sanity（phase 1+2 应基本相同）
for tag, d in (("baseline", base), ("model", model)):
    m = (d["phase"] == 1) | (d["phase"] == 2)
    print(f"excitation-phase check [{tag}]: peak|v|={np.abs(d['w'][m]).max():.3f} rad/s")

eb, tb, taub = tracking_metrics(base, "BASELINE (original controller)")
em, tm, taum = tracking_metrics(model, "MODEL (feedforward)")

# 直接逐样本误差对比（同轨迹）
print("\n===== head-to-head (model vs baseline) =====")
n = min(len(eb), len(em))
diff_rms = np.degrees(np.sqrt(np.mean(em[:n] ** 2)) - np.sqrt(np.mean(eb[:n] ** 2)))
print(f"RMS err: baseline={np.degrees(np.sqrt(np.nanmean(eb ** 2))):.3f}deg  "
      f"model={np.degrees(np.sqrt(np.nanmean(em ** 2))):.3f}deg")
print(f"peak |err|: baseline={np.degrees(np.nanmax(np.abs(eb))):.3f}deg  "
      f"model={np.degrees(np.nanmax(np.abs(em))):.3f}deg")
print(f"tau peak: baseline={np.nanmax(np.abs(taub)):.3f}  model={np.nanmax(np.abs(taum)):.3f} N*m")
print(f"tau smoothness(std diff): baseline={np.nanstd(np.diff(taub)) * 1000:.2f}  "
      f"model={np.nanstd(np.diff(taum)) * 1000:.2f} mN*m/ms")
ff = model["ff"]
ff3 = ff[(model["phase"] == 3) & ~np.isnan(ff)]
if len(ff3):
    print(f"model feedforward contribution: rms={np.sqrt(np.mean(ff3 ** 2)):.3f}  "
          f"peak={np.max(np.abs(ff3)):.3f} N*m")
