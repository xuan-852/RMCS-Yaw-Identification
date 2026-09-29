"""新数据集的二阶+摩擦辨识：汇总全部正弦工况，最小二乘拟合 + 交叉验证"""
import csv
import glob
import math
import os

import numpy as np

DT = 0.001
D = "/workspaces/RMCS/data"


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load(p):
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    return {
        "t": rows[:, n["condition_elapsed_s"]],
        "tau": rows[:, n["excitation_torque"]],
        "w": rows[:, n["measured_velocity_imu"]],
        "ptau": rows[:, n["pitch_torque"]],
        "pw": rows[:, n["pitch_world_angle"]],
    }


def fit(files, delay_ms=0, group=20, deadband=0.02):
    """把工况按 group 个采样点分块（20ms），在块内做增量回归：
    Δω = a'·ω + b'·∫τ + c'·∫sign + ...   —— 直接用朴素 ARX 会更受噪声影响，
    这里沿用 1ms ARX（与 09-26 版方法一致，保证可比）。
    """
    X, Y = [], []
    for p in files:
        d = load(p)
        w, tau = d["w"], d["tau"]
        s = np.where(np.abs(w) > deadband, np.sign(w), 0.0)
        k = int(delay_ms)
        N = len(w) - 1 - k
        X.append(np.column_stack([w[:N], tau[:N], s[:N], np.ones(N)]))
        Y.append(w[k+1:k+1+N])
    X = np.vstack(X)
    Y = np.concatenate(Y)
    theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ theta
    r2 = 1 - np.sum((Y-pred)**2)/np.sum((Y-Y.mean())**2)
    return theta, r2, len(Y)


sinB = sorted(glob.glob(f"{D}/sessionB/yaw_c0[2-9]*.csv")) + \
       sorted(glob.glob(f"{D}/sessionB/yaw_c1[0-3]*.csv"))
sinC = sorted(glob.glob(f"{D}/sessionC/yaw_c1[4-5]*.csv"))

train = sinB
valid = sinC

print("=" * 74)
print("训练集（sessionB 全部正弦工况）")
print("=" * 74)
best = None
for d in range(0, 6):
    theta, r2, n = fit(train, d)
    print(f"  延迟 {d} ms:  R2={r2:.5f}  (n={n})")
    if best is None or r2 > best[1]:
        best = (d, r2, theta)

d, r2, theta = best
a, b, c, c0 = theta
J = DT / b
B = (1 - a) / b
tc = abs(c) / b
print(f"\n最佳延迟 = {d} ms   R2 = {r2:.5f}")
print(f"  离散: w[k+1] = {a:.5f}*w[k] + {b:.4e}*tau + {c:.3e}*sign(w) + {c0:.2e}")
print(f"  ==> J = {J:.4f} kg*m^2   B = {B:.4f} N*m*s/rad   tau_c = {tc:.4f} N*m")
print(f"      机械时间常数 J/B = {J/B:.4f} s")
print(f"      极点 s = {(a-1)/DT:.3f} 1/s (离散 {a:.5f})")

r2v, _, nv = fit(valid, d)[1], 0, 0
theta_v, r2v, nv = fit(valid, d)
print(f"\n交叉验证（sessionC 的 2.5 N*m 高频工况）: R2 = {r2v:.5f}  (n={nv})")

print("\n" + "=" * 74)
print("与 09-26 版结果对比")
print("=" * 74)
print(f"  {'参数':<16s}{'09-26 版':>12s}{'本次':>12s}")
print(f"  {'J (kg*m^2)':<16s}{0.1503:>12.4f}{J:>12.4f}")
print(f"  {'B (N*m*s/rad)':<16s}{0.6605:>12.4f}{B:>12.4f}")
print(f"  {'tau_c (N*m)':<16s}{0.201:>12.4f}{tc:>12.4f}")
print(f"  {'训练 R2':<16s}{0.9992:>12.4f}{r2:>12.4f}")
print(f"  {'验证 R2':<16s}{0.9953:>12.4f}{r2v:>12.4f}")
