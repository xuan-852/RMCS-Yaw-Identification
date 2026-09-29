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
pw = np.degrees(rows[:, names["pitch_world_angle"]])
ptau = rows[:, names["pitch_torque"]]
ptemp = rows[:, names["pitch_temperature"]]
yvel = rows[:, names["measured_velocity_imu"]]
pang = rows[:, names["pitch_angle"]]

print("  t(s)   pitch_world(deg)  pitch_motor(deg)  tau_pitch(N*m)  pitch_T(C)  yaw_vel")
for i in range(0, len(t), 1000):  # 每秒一行
    print(f"  {t[i]:5.1f}   {pw[i]:+8.1f}        {math.degrees(pang[i]):8.1f}        "
          f"{ptau[i]:+7.3f}       {ptemp[i]:5.0f}      {yvel[i]:+6.2f}")
