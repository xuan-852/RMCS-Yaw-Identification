import csv
import glob
import math
import os

import numpy as np

D = "/workspaces/RMCS/data"
FILES = sorted(glob.glob(f"{D}/sessionB/*.csv")) + sorted(glob.glob(f"{D}/sessionC/*.csv"))


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


print(f"{'工况':18s} {'时长s':>6s} {'行数':>7s} {'采样率':>7s} {'|w|峰':>6s} "
      f"{'偏转°':>6s} {'|tau|max':>8s} {'pitch动°':>8s} {'温度':>8s} {'判定':>6s}")
print("-" * 92)

issues = []
for p in FILES:
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    t = rows[:, n["condition_elapsed_s"]]
    w = rows[:, n["measured_velocity_imu"]]
    tau = rows[:, n["measured_torque"]]
    yaw = np.degrees(rows[:, n["measured_angle"]])
    pw = np.degrees(rows[:, n["pitch_world_angle"]])
    pt = rows[:, n["pitch_temperature"]]
    yaw_u = np.zeros(len(yaw))
    for i in range(1, len(yaw)):
        yaw_u[i] = yaw_u[i-1] + np.remainder(yaw[i]-yaw[i-1]+180, 360) - 180

    dur = t[-1]
    rate = len(t) / dur if dur > 0 else 0
    exc = yaw_u.max() - yaw_u.min()
    dtau = np.abs(np.diff(tau)) / np.maximum(np.diff(t), 1e-9)
    dpitch = pw.max() - pw.min()

    ok = "OK"
    if abs(rate - 1000) > 5:
        ok = "采样异常"; issues.append((os.path.basename(p), "采样率"))
    if np.abs(w).max() > 9.5:
        ok = "近速度限"; issues.append((os.path.basename(p), "速度"))
    if dpitch > 5:
        ok = "pitch动"; issues.append((os.path.basename(p), f"pitch{dpitch:.0f}deg"))
    if pt.max() >= 68:
        ok = "近温度限"; issues.append((os.path.basename(p), "温度"))

    name = os.path.basename(p).split("_2026")[0]
    print(f"{name:18s} {dur:6.1f} {len(t):7d} {rate:7.0f} {np.abs(w).max():6.2f} "
          f"{exc:6.1f} {np.abs(tau).max():8.2f} {dpitch:8.2f} "
          f"{pt[0]:3.0f}->{pt[-1]:3.0f} {ok:>6s}")

print("-" * 92)
print(f"\n总工况数: {len(FILES)}")
if issues:
    print(f"\n需要注意的条目 ({len(issues)}):")
    for f, why in issues:
        print(f"  {f}  -> {why}")
else:
    print("\n全部通过 ✓")

# 覆盖度统计
print(f"\n=== 覆盖度 ===")
peaks = []
for p in FILES:
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    w = rows[:, n["measured_velocity_imu"]]
    peaks.append(np.abs(w).max())
print(f"  各工况峰值角速度范围: {min(peaks):.2f} ~ {max(peaks):.2f} rad/s")
