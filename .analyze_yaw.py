import csv
import math
import os
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspaces/RMCS/data/" + os.listdir("/workspaces/RMCS/data")[0]

with open(PATH) as f:
    reader = csv.reader(f)
    header = next(reader)
    parsed = ([float(x) if x not in ("", "nan", "-nan") else math.nan for x in row] for row in reader)
    rows = np.array([r + [math.nan] * (len(header) - len(r)) for r in parsed])

names = {n: i for i, n in enumerate(header)}
col = lambda n: rows[:, names[n]]
t = col("elapsed_s")
phase = col("phase")
tau_ex = col("excitation_torque")
tau_fb = col("measured_torque")
w_mot = col("measured_velocity")
w_imu = col("measured_velocity_imu")
ang = col("measured_angle")

print(f"file={PATH.split('/')[-1]}  rows={len(rows)}  duration={t[-1] - t[0]:.1f}s")

print("\n=== phase boundaries ===")
for pid, name in ((0, "static"), (1, "sine"), (2, "step")):
    m = phase == pid
    if m.any():
        i = np.where(m)[0]
        print(f"phase {pid} ({name}): rows={m.sum()} t=[{t[i[0]]:.2f} ~ {t[i[-1]]:.2f}]s")

print("\n=== static phase (zero bias) ===")
m = phase == 0
if m.sum() > 100:
    print(f"imu_vel  bias/std: {w_imu[m].mean():+.4f} / {w_imu[m].std():.4f} rad/s")
    print(f"motor_vel bias/std: {w_mot[m].mean():+.4f} / {w_mot[m].std():.4f} rad/s")
    print(f"tau_fb   bias/std: {tau_fb[m].mean():+.4f} / {tau_fb[m].std():.4f} N*m")

print("\n=== sine phase response ===")
m = (phase == 1) & (t > 20)  # 跳过最初5s过渡
if m.sum() > 100:
    print(f"excitation:      min={tau_ex[m].min():+.3f} max={tau_ex[m].max():+.3f} N*m")
    print(f"measured torque: min={tau_fb[m].min():+.3f} max={tau_fb[m].max():+.3f} N*m")
    print(f"imu velocity:    min={w_imu[m].min():+.3f} max={w_imu[m].max():+.3f} rad/s  peak|v|={np.abs(w_imu[m]).max():.3f}")
    seg = ang[m]
    print(f"angle swing in sine: {(seg.max() - seg.min()):.3f} rad = {np.degrees(seg.max() - seg.min()):.1f} deg")
    # 响应与激励的基频相关（粗查相位滞后）
    exc = tau_ex[m] - tau_ex[m].mean()
    vel = w_imu[m] - w_imu[m].mean()
    xc = np.correlate(vel - vel.mean(), exc - exc.mean(), "full")
    lag = np.argmax(xc) - (len(exc) - 1)
    print(f"velocity-vs-torque cross-corr peak lag: {lag} samples = {lag} ms")

print("\n=== step phase response ===")
m = phase == 2
if m.sum() > 100:
    print(f"excitation: min={tau_ex[m].min():+.3f} max={tau_ex[m].max():+.3f} N*m")
    print(f"imu velocity: min={w_imu[m].min():+.3f} max={w_imu[m].max():+.3f} rad/s")

print("\n=== consistency & saturation ===")
mm = ~np.isnan(w_imu) & ~np.isnan(w_mot)
slope = np.polyfit(w_imu[mm], w_mot[mm], 1)
print(f"motor_vel = {slope[0]:.3f} * imu_vel + {slope[1]:+.3f}  (corr={np.corrcoef(w_imu[mm], w_mot[mm])[0,1]:.4f})")
print(f"|measured_torque| max = {np.abs(tau_fb).max():.3f} N*m (电机上限4.5, 激励限幅1.5)")
print(f"excitation clipped rows(|exc|==0.8 or sine clip): {(np.abs(tau_ex) > 1.49).sum()}")
