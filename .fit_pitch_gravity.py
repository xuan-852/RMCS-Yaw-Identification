"""pitch 重力补偿标定分析
拟合 tau_gravity(p) = A*cos(p) + B*sin(p) + C，与现有前馈参数对比。
"""
import csv
import math
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspaces/RMCS/data/diag/c01.csv"


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
ptau = rows[:, names["pitch_torque"]]
pw = rows[:, names["pitch_world_angle"]]
ptemp = rows[:, names["pitch_temperature"]]
yvel = rows[:, names["measured_velocity_imu"]]

print(f"rows={len(t)} duration={t[-1]:.1f}s")
print(f"pitch 世界角范围: {np.degrees(np.nanmin(pw)):+.1f} ~ {np.degrees(np.nanmax(pw)):+.1f} deg")
print(f"pitch 实测力矩范围: {np.nanmin(ptau):+.3f} ~ {np.nanmax(ptau):+.3f} N*m")
print(f"pitch 温度: {np.nanmin(ptemp):.0f} ~ {np.nanmax(ptemp):.0f} C")

# 选稳态窗口：pitch 角速度很小 + 力矩平稳
dp = np.abs(np.gradient(pw, t))
dtau = np.abs(np.gradient(ptau, t))
steady = (dp < np.radians(1.5)) & (dtau < 0.5) & np.isfinite(pw) & np.isfinite(ptau)
print(f"\n稳态样本: {steady.sum()} / {len(t)} ({100*steady.mean():.1f}%)")

p = pw[steady]
tau = ptau[steady]
print(f"稳态 pitch 角范围: {np.degrees(p.min()):+.1f} ~ {np.degrees(p.max()):+.1f} deg")
print(f"稳态力矩范围: {tau.min():+.3f} ~ {tau.max():+.3f} N*m")

# 按角度分箱看原始关系
print("\n=== 分箱平均（角度 -> 保持力矩）===")
bins = np.linspace(p.min(), p.max(), 12)
for i in range(len(bins) - 1):
    m = (p >= bins[i]) & (p < bins[i + 1])
    if m.sum() > 50:
        print(f"  pitch={np.degrees(0.5*(bins[i]+bins[i+1])):+6.1f} deg  "
              f"tau={tau[m].mean():+.3f} N*m  (n={m.sum()})")

# 最小二乘拟合
X = np.column_stack([np.cos(p), np.sin(p), np.ones(len(p))])
theta, *_ = np.linalg.lstsq(X, tau, rcond=None)
A, B, C = theta
pred = X @ theta
r2 = 1 - np.sum((tau - pred) ** 2) / np.sum((tau - tau.mean()) ** 2)
print(f"\n=== 拟合结果 tau = A*cos(p) + B*sin(p) + C ===")
print(f"  A = {A:+.4f}   B = {B:+.4f}   C = {C:+.4f}")
print(f"  R2 = {r2:.4f}   残差 std = {np.std(tau - pred):.4f} N*m  振幅 = {math.hypot(A,B):.4f} N*m")

# 现有前馈参数
G, PH = 2.575, 1.784
A_ff = -G * math.sin(PH)
B_ff = G * math.cos(PH)
print(f"\n=== 现有前馈 pitch_gravity_ff_gain={G}, phase={PH} ===")
print(f"  等效 A_ff = {A_ff:+.4f}   B_ff = {B_ff:+.4f}")
print(f"  前馈振幅 = {G:.4f} N*m")
print(f"\n=== 对比 ===")
print(f"  A: 拟合 {A:+.4f}  vs 前馈 {A_ff:+.4f}   差 {A - A_ff:+.4f}")
print(f"  B: 拟合 {B:+.4f}  vs 前馈 {B_ff:+.4f}   差 {B - B_ff:+.4f}")
print(f"  振幅比 (拟合/前馈) = {math.hypot(A,B)/G:.3f}")
print(f"  C(常数偏置) = {C:+.4f} N*m")

# 推荐新参数：保持现有相位，只改增益；或重新解出 gain/phase
G_new = math.hypot(A, B)
PH_new = math.atan2(-A, B)  # 使 A=-G*sin(PH), B=G*cos(PH)
print(f"\n=== 建议新参数（按拟合振幅/相位）===")
print(f"  pitch_gravity_ff_gain: {G_new:.3f}")
print(f"  pitch_gravity_ff_phase: {PH_new:.3f}")
pred_ff = G_new * np.sin(p - PH_new)
print(f"  用新参数覆盖稳态力矩的残差 std = {np.std(tau - pred_ff):.4f} N*m")
pred_old = G * np.sin(p - PH)
print(f"  用旧参数的残差 std = {np.std(tau - pred_old):.4f} N*m  "
      f"(旧参数欠/过补偿均值 {np.mean(tau - pred_old):+.3f} N*m)")
