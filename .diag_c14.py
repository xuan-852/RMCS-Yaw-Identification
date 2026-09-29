import csv
import math
import sys

import numpy as np

PATH = sys.argv[1]


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


with open(PATH) as f:
    r = csv.reader(f)
    h = next(r)
    names = {n: i for i, n in enumerate(h)}
    rows = np.array([[_f(x) for x in row] for row in r])

t = rows[:, names["condition_elapsed_s"]]
exc = rows[:, names["excitation_torque"]]
yaw = np.degrees(rows[:, names["measured_angle"]])
yvel = rows[:, names["measured_velocity_imu"]]
ptau = rows[:, names["pitch_torque"]]
pw = np.degrees(rows[:, names["pitch_world_angle"]])
ptemp = rows[:, names["pitch_temperature"]]

yaw_u = np.zeros(len(yaw))
for i in range(1, len(yaw)):
    yaw_u[i] = yaw_u[i-1] + np.remainder(yaw[i]-yaw[i-1]+180, 360) - 180

print(f"时长={t[-1]:.2f}s   工况 excitation 峰值={np.abs(exc).max():.2f} N*m")
print(f"\n   t(s)   yaw_exc  yaw_vel  yaw累计(deg)  pitch_world  pitch_tau   pitch_T")
step = max(1, len(t)//40)
for i in range(0, len(t), step):
    print(f"  {t[i]:6.2f}   {exc[i]:+7.2f}  {yvel[i]:+7.2f}  {yaw_u[i]-yaw_u[0]:+9.1f}   "
          f"{pw[i]:+8.2f}   {ptau[i]:+8.3f}   {ptemp[i]:5.0f}")

drop = np.abs(np.diff(pw))
idx = np.where(drop > 0.5)[0]
print(f"\n=== pitch 掉落起点 ===")
if len(idx):
    i = idx[0]
    print(f"  t={t[i]:.2f}s  此时 yaw 速度={yvel[i]:+.2f} rad/s  yaw 累计={yaw_u[i]-yaw_u[0]:+.1f} deg")
    print(f"  pitch 世界角: {pw[i]:+.2f} -> {pw[i+1]:+.2f} deg")
    print(f"  pitch 力矩:   {ptau[i]:+.3f} -> {ptau[i+1]:+.3f} N*m")
    print(f"\n  掉落前 0.3s 内 yaw 速度: 均值={np.mean(np.abs(yvel[max(0,i-300):i])):.2f} "
          f"峰值={np.max(np.abs(yvel[max(0,i-300):i])):.2f} rad/s")
    print(f"  掉落全程 pitch 世界角变化: {pw[i:].max()-pw[i:].min():.2f} deg")
else:
    print("  未检测到 >0.5deg 的突变")
    print(f"  全程 pitch 世界角变化 {pw.max()-pw.min():.2f} deg")

print(f"\n=== yaw 速度峰值 ===")
print(f"  峰值 |v| = {np.abs(yvel).max():.2f} rad/s")
print(f"  |v|>3 rad/s 的样本数: {np.sum(np.abs(yvel)>3)}")
print(f"  |v|>5 rad/s 的样本数: {np.sum(np.abs(yvel)>5)}")
