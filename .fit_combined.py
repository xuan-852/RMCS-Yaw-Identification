"""合并拟合：09-26 低频数据（提供 tau_c 信息）+ 09-29 全量高频数据（提供 B 信息）"""
import csv
import glob
import math

import numpy as np

DT = 0.001


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load_any(p):
    """兼容两种 CSV 列名格式"""
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    tkey = "condition_elapsed_s" if "condition_elapsed_s" in n else "elapsed_s"
    tau = rows[:, n["excitation_torque"]]
    w = rows[:, n["measured_velocity_imu"]]
    t = rows[:, n[tkey]]
    m = np.isfinite(tau) & np.isfinite(w) & (t >= 0.5)
    return tau[m], w[m]


def fit(files, delay_ms=0, deadband=0.02):
    X, Y = [], []
    for p in files:
        tau, w = load_any(p)
        s = np.where(np.abs(w) > deadband, np.sign(w), 0.0)
        k = int(delay_ms)
        N = len(w) - 1 - k
        X.append(np.column_stack([w[:N], tau[:N], s[:N], np.ones(N)]))
        Y.append(w[k+1:k+1+N])
    X = np.vstack(X); Y = np.concatenate(Y)
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ theta
    r2 = 1 - np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
    return theta, r2, len(Y)


def report(tag, theta, r2, n):
    a, b, c, c0 = theta
    J = DT/b; B = (1-a)/b; tc = abs(c)/b
    print(f"{tag}")
    print(f"  n={n:8d}   R2={r2:.5f}")
    print(f"  J={J:.4f} kg*m^2   B={B:.4f} N*m*s/rad   tau_c={tc:.4f} N*m   "
          f"(J/B={J/B:.3f}s)")
    return J, B, tc


old = ["/workspaces/RMCS/data/yaw_identification_2026-09-26_09-47-20.csv",
       "/workspaces/RMCS/data/yaw_identification_2026-09-26_09-52-07.csv"]
new_sin = sorted(glob.glob("/workspaces/RMCS/data/sessionB/yaw_c0[2-9]*.csv")) + \
          sorted(glob.glob("/workspaces/RMCS/data/sessionB/yaw_c1[0-3]*.csv")) + \
          sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c1[4-5]*.csv"))
new_step = sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c1[89]*.csv")) + \
           sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c20*.csv"))
new_trk = sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c2[1-6]*.csv"))

print("=" * 74)
print("A) 仅 09-26 数据（原报告用的数据集）")
print("=" * 74)
report("  ", *fit(old, 0))

print("\n" + "=" * 74)
print("B) 仅 09-29 新的正弦数据（无低频段）")
print("=" * 74)
report("  ", *fit(new_sin, 0))

print("\n" + "=" * 74)
print("C) 09-26 低频 + 09-29 正弦（合并）")
print("=" * 74)
report("  ", *fit(old + new_sin, 0))

print("\n" + "=" * 74)
print("D) 合并 + 阶跃（阶跃含大量过零，对 tau_c 有利）")
print("=" * 74)
report("  ", *fit(old + new_sin + new_step, 0))

print("\n" + "=" * 74)
print("E) 全部（含闭环 track，输入为控制器指令）")
print("=" * 74)
report("  ", *fit(old + new_sin + new_step + new_trk, 0))

print("\n" + "=" * 74)
print("物理自检：用各组参数预测 0.5 Hz 扫幅的峰值角速度 (tau-tau_c)/B")
print("=" * 74)
print(f"  {'幅值':>7s} {'实测峰值':>9s}   A(09-26)   B(新)    C(合并)")
tests = [(0.3, 0.11), (0.6, 0.29), (1.5, 2.28), (2.5, 2.72)]
params = [fit(old, 0)[0], fit(new_sin, 0)[0], fit(old+new_sin, 0)[0]]
pars = []
for th in params:
    a, b, c, c0 = th
    pars.append((DT/b, (1-a)/b, abs(c)/b))
for amp, meas in tests:
    row = f"  {amp:7.1f} {meas:9.2f}  "
    for J, B, tc in pars:
        pred = max(0.0, (amp-tc)/B)
        row += f"{pred:9.2f}"
    print(row)
