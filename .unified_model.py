"""统一模型辨识 + 自然遥控数据拟合度评估

模型: J*dw/dt = tau_applied - B*w - tau_c*sign(w)
- 输入用【实测作用力矩 measured_torque】(电调电流换算)，而非控制器指令
- 拟合: 一步 ARX 求初值 -> 坐标下降最小化"自由仿真误差"(输出误差法)
- 验证: 未参与拟合的工况 + 自然遥控数据
"""
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
    tk = "condition_elapsed_s" if "condition_elapsed_s" in n else "elapsed_s"
    t = rows[:, n[tk]]
    tau = rows[:, n["measured_torque"]]      # 实测作用力矩
    w = rows[:, n["measured_velocity_imu"]]
    m = np.isfinite(tau) & np.isfinite(w) & (t >= 0.3)
    return tau[m], w[m]


def simulate(tau, w0, J, B, tc):
    n = len(tau)
    w = np.empty(n)
    w[0] = w0
    for i in range(n - 1):
        s = 1.0 if w[i] > 0 else (-1.0 if w[i] < 0 else 0.0)
        w[i+1] = w[i] + DT * (tau[i] - B * w[i] - tc * s) / J
    return w


def sim_error(files, J, B, tc):
    """各工况自由仿真误差（归一化 RMS）"""
    errs, corrs = [], []
    for p in files:
        tau, w = load(p)
        sim = simulate(tau, w[0], J, B, tc)
        e = np.sqrt(np.mean((sim - w) ** 2)) / max(np.std(w), 1e-6)
        errs.append(e)
        corrs.append(np.corrcoef(w, sim)[0, 1] if np.std(w) > 1e-9 else 0.0)
    return np.mean(errs), errs, corrs


# ---------- 训练/验证划分 ----------
train = ["/workspaces/RMCS/data/yaw_identification_2026-09-26_09-52-07.csv",
         "/workspaces/RMCS/data/yaw_identification_2026-09-26_09-47-20.csv"] + \
        sorted(glob.glob(f"{D}/sessionB/yaw_c0[2-9]*.csv")) + \
        sorted(glob.glob(f"{D}/sessionB/yaw_c1[0-3]*.csv")) + \
        sorted(glob.glob(f"{D}/sessionC/yaw_c1[4-5]*.csv"))
holdout = sorted(glob.glob(f"{D}/sessionC/yaw_c1[89]*.csv")) + \
          sorted(glob.glob(f"{D}/sessionC/yaw_c2[0-6]*.csv"))
rc = sorted(glob.glob(f"{D}/sessionC/yaw_c27_rc*.csv"))

print("=" * 76)
print("第一步：一步 ARX 求初值（输入 = 实测作用力矩）")
print("=" * 76)
X, Y = [], []
for p in train:
    tau, w = load(p)
    s = np.where(np.abs(w) > 0.02, np.sign(w), 0.0)
    N = len(w) - 1
    X.append(np.column_stack([w[:N], tau[:N], s[:N], np.ones(N)]))
    Y.append(w[1:1+N])
X = np.vstack(X); Y = np.concatenate(Y)
theta, *_ = np.linalg.lstsq(X, Y, rcond=None)
a, b, c, c0 = theta
J0, B0, tc0 = DT/b, (1-a)/b, abs(c)/b
r2 = 1 - np.sum((Y - X@theta)**2)/np.sum((Y-Y.mean())**2)
print(f"  J={J0:.4f}  B={B0:.4f}  tau_c={tc0:.4f}   (一步 R2={r2:.5f}, n={len(Y)})")

print("\n" + "=" * 76)
print("第二步：坐标下降最小化【自由仿真误差】（输出误差法，物理意义上更正确）")
print("=" * 76)


def cost(J, B, tc):
    if J <= 0.02 or B <= 0.0 or tc < 0.0:
        return 1e9
    return sim_error(train, J, B, tc)[0]


J, B, tc = J0, B0, tc0
best = cost(J, B, tc)
print(f"  初值代价 = {best:.5f}")
for it in range(40):
    improved = False
    for name, val, step in (("J", J, 0.02), ("B", B, 0.15), ("tc", tc, 0.08)):
        for sgn in (+1, -1):
            if name == "J":   cand = (val + sgn*step, B, tc)
            elif name == "B": cand = (J, val + sgn*step, tc)
            else:             cand = (J, B, val + sgn*step)
            c_ = cost(*cand)
            if c_ < best - 1e-6:
                best, (J, B, tc) = c_, cand
                improved = True
    if not improved:
        break
print(f"  收敛: J={J:.4f}  B={B:.4f}  tau_c={tc:.4f}   代价={best:.5f}")
print(f"  机械时间常数 J/B = {J/B:.3f} s")
print(f"  转折频率 B/(2*pi*J) = {B/(2*math.pi*J):.3f} Hz")

print("\n" + "=" * 76)
print("第三步：统一模型在两批数据上的表现（自由仿真）")
print("=" * 76)
for tag, files in (("训练集(09-26低频 + 09-29正弦)", train),
                   ("留出集(阶跃 + A/B 跟踪)", holdout)):
    m, errs, corrs = sim_error(files, J, B, tc)
    print(f"\n  【{tag}】{len(files)} 条工况")
    print(f"    归一化 RMS 误差: 均值 {m:.3f}  中位 {np.median(errs):.3f}  最大 {np.max(errs):.3f}")
    print(f"    波形相关系数:   均值 {np.mean(corrs):.3f}")

print("\n" + "=" * 76)
print("第四步：★ 拟合自然遥控控制数据 ★")
print("=" * 76)
for p in rc:
    tau, w = load(p)
    sim = simulate(tau, w[0], J, B, tc)
    rms = np.sqrt(np.mean((sim-w)**2))
    rms_n = rms / max(np.std(w), 1e-6)
    corr = np.corrcoef(w, sim)[0, 1]
    print(f"\n  {os.path.basename(p)}")
    print(f"    时长 {len(w)*DT:.1f}s   实测 |w| 峰 {np.abs(w).max():.2f} rad/s   "
          f"作用力矩 |tau| 峰 {np.abs(tau).max():.2f} N*m")
    print(f"    自由仿真 vs 实测:  RMS 误差 = {rms:.4f} rad/s ({rms_n*100:.1f}% of std)")
    print(f"    波形相关系数 = {corr:.4f}")
    print(f"    实测 w 标准差 {np.std(w):.3f}  vs  仿真 w 标准差 {np.std(sim):.3f}")
