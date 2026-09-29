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
yaw = np.degrees(rows[:, names["measured_angle"]])
yvel = rows[:, names["measured_velocity_imu"]]
ptau = rows[:, names["pitch_torque"]]
pw = np.degrees(rows[:, names["pitch_world_angle"]])
ptemp = rows[:, names["pitch_temperature"]]

# 展开 yaw 角
yaw_u = np.zeros(len(yaw))
for i in range(1, len(yaw)):
    yaw_u[i] = yaw_u[i-1] + np.remainder(yaw[i] - yaw[i-1] + 180, 360) - 180

print(f"行数={len(t)}  时长={t[-1]:.1f}s")
print(f"\n=== yaw ===")
print(f"  累计转角: {yaw_u[-1]-yaw_u[0]:+.1f} deg")
print(f"  转速范围: {yvel.min():+.2f} ~ {yvel.max():+.2f} rad/s")
print(f"  转速 |v|>0.3 rad/s 的时间占比: {100*np.mean(np.abs(yvel)>0.3):.1f}%")

print(f"\n=== pitch（关键）===")
print(f"  世界角范围: {pw.min():+.2f} ~ {pw.max():+.2f} deg   (总变化 {pw.max()-pw.min():.2f} deg)")
print(f"  力矩范围:   {ptau.min():+.3f} ~ {ptau.max():+.3f} N*m")
print(f"  温度: {ptemp[0]:.0f} -> {ptemp[-1]:.0f} C")

print(f"\n=== 时间序列（每 1.5 秒）===")
print("   t(s)   yaw累计(deg)  yaw转速  pitch_world  pitch_tau  pitch_T")
for i in range(0, len(t), 1500):
    print(f"  {t[i]:6.1f}   {yaw_u[i]-yaw_u[0]:+9.1f}   {yvel[i]:+6.2f}   {pw[i]:+9.2f}   "
          f"{ptau[i]:+8.3f}   {ptemp[i]:5.0f}")

# 检查 yaw 转速是否有异常下降（干涉迹象）
print(f"\n=== 转速异常检查 ===")
moving = np.abs(yvel) > 0.3
if moving.sum() > 100:
    v_moving = np.abs(yvel[moving])
    print(f"  转动期间 |v| 均值={v_moving.mean():.2f} 最小={v_moving.min():.2f} rad/s")
    print(f"  转动期间 |v|<0.5 rad/s 的占比: {100*np.mean(v_moving<0.5):.1f}%  （高=有阻力/干涉）")

print(f"\n=== pitch 掉落检查 ===")
drop = pw.max() - pw.min()
print(f"  全程 pitch 世界角变化 {drop:.2f} deg  → {'★ 有掉落' if drop > 3 else '✓ 无掉落'}")
