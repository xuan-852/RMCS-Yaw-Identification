"""决定性验证：用 09-26 标定的参数自由仿真 09-29 的每一条正弦工况，比较峰值角速度"""
import csv
import glob
import math

import numpy as np

# 09-26 标定值（原报告）
J, B, TAU_C = 0.1503, 0.6605, 0.201
DT = 0.001


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load(p):
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    tk = "condition_elapsed_s" if "condition_elapsed_s" in n else "elapsed_s"
    return rows[:, n[tk]], rows[:, n["excitation_torque"]], rows[:, n["measured_velocity_imu"]]


def simulate(tau, w0):
    """J dw/dt + B w + tau_c sign(w) = tau  （欧拉，dt=1ms）"""
    w = np.empty(len(tau))
    w[0] = w0
    for i in range(len(tau) - 1):
        s = 1.0 if w[i] > 0 else (-1.0 if w[i] < 0 else 0.0)
        w[i+1] = w[i] + DT * (tau[i] - B*w[i] - TAU_C*s) / J
    return w


FILES = sorted(glob.glob("/workspaces/RMCS/data/sessionB/yaw_c0[2-9]*.csv")) + \
        sorted(glob.glob("/workspaces/RMCS/data/sessionB/yaw_c1[0-3]*.csv")) + \
        sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c1[4-5]*.csv"))

print(f"模型: J={J}  B={B}  tau_c={TAU_C}（09-26 标定值）")
print(f"{'工况':<14s}{'实测|w|峰':>10s}{'仿真|w|峰':>10s}{'误差':>9s}{'波形相关':>10s}")
print("-" * 56)
errs, cors = [], []
for p in FILES:
    t, tau, w = load(p)
    sim = simulate(tau, w[0])
    pm, ps = np.abs(w).max(), np.abs(sim).max()
    err = (ps - pm) / max(pm, 1e-6) * 100
    c = np.corrcoef(w, sim)[0, 1]
    errs.append(err); cors.append(c)
    print(f"{p.split('/')[-1].split('_2026')[0]:<14s}{pm:10.2f}{ps:10.2f}{err:+8.1f}%{c:10.3f}")
print("-" * 56)
print(f"峰值误差: 均值 {np.mean(errs):+.1f}%  中位 {np.median(errs):+.1f}%  最大 {np.max(np.abs(errs)):.1f}%")
print(f"波形相关系数: 均值 {np.mean(cors):.3f}  最小 {np.min(cors):.3f}")
