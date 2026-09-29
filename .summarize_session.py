import csv
import glob
import math
import os

import numpy as np

D = "/workspaces/RMCS/data/sessionC"


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


print(f"{'工况':22s} {'时长s':>6s} {'|w|峰':>7s} {'偏转°':>7s} {'pitch角变化°':>11s} {'pitch_T':>8s}")
print("-" * 70)
for p in sorted(glob.glob(f"{D}/*.csv")):
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    t = rows[:, n["condition_elapsed_s"]]
    w = rows[:, n["measured_velocity_imu"]]
    yaw = np.degrees(rows[:, n["measured_angle"]])
    pw = np.degrees(rows[:, n["pitch_world_angle"]])
    pt = rows[:, n["pitch_temperature"]]
    yaw_u = np.zeros(len(yaw))
    for i in range(1, len(yaw)):
        yaw_u[i] = yaw_u[i-1] + np.remainder(yaw[i]-yaw[i-1]+180, 360) - 180
    exc = yaw_u.max() - yaw_u.min()
    name = os.path.basename(p).split("_2026")[0]
    print(f"{name:22s} {t[-1]:6.1f} {np.abs(w).max():7.2f} {exc:7.1f} "
          f"{pw.max()-pw.min():11.2f} {pt[0]:4.0f}->{pt[-1]:.0f} C")
