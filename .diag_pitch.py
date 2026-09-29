import csv
import math
import os

import numpy as np

D = "/workspaces/RMCS/data/diag"


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load(p):
    with open(p) as f:
        r = csv.reader(f)
        h = next(r)
        names = {n: i for i, n in enumerate(h)}
        rows = np.array([[_f(x) for x in row] for row in r])
    return {n: rows[:, i] for n, i in names.items()}


for fn in sorted(os.listdir(D)):
    d = load(os.path.join(D, fn))
    t = d["condition_elapsed_s"]
    print("=" * 64)
    print(fn)
    print(f"  rows={len(t)} duration={t[-1]:.2f}s  condition={int(d['condition_id'][0])} "
          f"({int(d['phase'][0])})")
    pa = np.degrees(d["pitch_angle"])
    print(f"  pitch_angle: start={pa[0]:+.1f} end={pa[-1]:+.1f} min={pa.min():+.1f} max={pa.max():+.1f} deg")
    # 找 pitch 的大幅变化点
    dpa = np.abs(np.diff(pa))
    if dpa.max() > 1.0:
        idx = np.where(dpa > 1.0)[0]
        print(f"  pitch 变化 >1deg 的样本数: {len(idx)}")
        for i in idx[:5]:
            print(f"    t={t[i]:.2f}s  {pa[i]:+.1f} -> {pa[i+1]:+.1f} deg  (d={dpa[i]:.1f})")
    else:
        print("  pitch 全程变化 <1deg（未掉落）")
    w = d["measured_velocity_imu"]
    print(f"  yaw vel(imu): peak={np.abs(w).max():.2f} rad/s 末值={w[-1]:+.2f}")
    print(f"  temperature: {np.nanmin(d['temperature']):.0f} -> {np.nanmax(d['temperature']):.0f} C")
    print(f"  末 3 行 (t, exc, tau_meas, vel_imu, pitch):")
    for i in range(-3, 0):
        print(f"    t={t[i]:.2f} exc={d['excitation_torque'][i]:+.3f} "
              f"tau_meas={d['measured_torque'][i]:+.3f} vel={w[i]:+.3f} pitch={pa[i]:+.1f}deg")
