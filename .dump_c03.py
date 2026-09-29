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
yvel = rows[:, names["measured_velocity_imu"]]
ptau = rows[:, names["pitch_torque"]]
pw = np.degrees(rows[:, names["pitch_world_angle"]])
ptemp = rows[:, names["pitch_temperature"]]
pmot = np.degrees(rows[:, names["pitch_angle"]])

print("   t(s)   yaw_exc  yaw_vel   pitch_tau  pitch_world  pitch_motor  pitch_T")
step = max(1, len(t) // 60)
for i in range(0, len(t), step):
    print(f"  {t[i]:6.2f}  {exc[i]:+7.3f}  {yvel[i]:+7.3f}   {ptau[i]:+8.3f}   {pw[i]:+8.2f}    "
          f"{pmot[i]:8.1f}    {ptemp[i]:5.0f}")

print(f"\n=== pitch_torque 绝对值跌破 1 N*m 的第一处 ===")
idx = np.where(np.abs(ptau) < 1.0)[0]
if len(idx):
    i = idx[0]
    print(f"  t={t[i]:.3f}s  pitch_tau {ptau[i]:+.3f}  (前 0.1s: "
          f"{ptau[max(0,i-100)]:+.3f})  pitch_world {pw[i]:+.2f} deg")
else:
    print("  未出现")
print(f"\n=== pitch_world_angle 变化最大的瞬间 ===")
d = np.abs(np.diff(pw)) / np.maximum(np.diff(t), 1e-9)
top = np.argsort(d)[::-1][:5]
for i in sorted(top):
    print(f"  t={t[i]:.3f}s  rate={d[i]:.1f} deg/s  pitch {pw[i]:+.2f} -> {pw[i+1]:+.2f}  "
          f"tau {ptau[i]:+.3f} -> {ptau[i+1]:+.3f}")
