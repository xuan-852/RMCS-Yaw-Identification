"""统一模型：三维网格搜索 + 细化，目标 = 自由仿真误差（输出误差法）
模型: J*dw/dt = tau_measured - B*w - tau_c*sign(w)
"""
import csv
import glob
import math
import os

import numpy as np

DT = 0.001
STEP = 4                      # 仿真步长 4ms（动力学 ~0.3s，足够）
D = "/workspaces/RMCS/data"


def _f(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load(p, dec=STEP):
    rows = np.array([[_f(x) for x in r] for r in list(csv.reader(open(p)))[1:]])
    h = list(csv.reader(open(p)))[0]
    n = {k: i for i, k in enumerate(h)}
    tk = "condition_elapsed_s" if "condition_elapsed_s" in n else "elapsed_s"
    t = rows[:, n[tk]]
    tau = rows[:, n["measured_torque"]]
    w = rows[:, n["measured_velocity_imu"]]
    m = np.isfinite(tau) & np.isfinite(w) & (t >= 0.3)
    return tau[m][::dec], w[m][::dec]


def simulate(tau, w0, J, B, tc, dt):
    w = np.empty(len(tau)); w[0] = w0
    for i in range(len(tau)-1):
        s = 1.0 if w[i] > 0 else (-1.0 if w[i] < 0 else 0.0)
        w[i+1] = w[i] + dt*(tau[i] - B*w[i] - tc*s)/J
    return w


# 训练集：挑最有信息量的（低频 09-26 + 09-29 频率扫描 + 小幅值）
train = ["/workspaces/RMCS/data/yaw_identification_2026-09-26_09-52-07.csv",
         "/workspaces/RMCS/data/yaw_identification_2026-09-26_09-47-20.csv"] + \
        sorted(glob.glob(f"{D}/sessionB/yaw_c0[2-9]*.csv"))[:6]
val = sorted(glob.glob(f"{D}/sessionC/yaw_c1[4-5]*.csv")) + \
      sorted(glob.glob(f"{D}/sessionC/yaw_c1[89]*.csv"))
rc = sorted(glob.glob(f"{D}/sessionC/yaw_c27_rc*.csv"))

cache = {p: load(p) for p in set(train+val+rc)}
DT_S = DT*STEP


def cost(J, B, tc, files):
    tot = 0.0
    for p in files:
        tau, w = cache[p]
        sim = simulate(tau, w[0], J, B, tc, DT_S)
        sd = np.std(w)
        tot += np.sqrt(np.mean((sim-w)**2))/max(sd, 1e-6)
    return tot/len(files)


print("="*76)
print("三维网格搜索（目标：自由仿真归一化误差）")
print("="*76)
best = None
for J in (0.12, 0.14, 0.15, 0.16, 0.18):
    for B in (0.20, 0.35, 0.50, 0.66, 0.80, 0.95):
        for tc in (0.05, 0.12, 0.20, 0.28, 0.38, 0.48):
            c = cost(J, B, tc, train)
            if best is None or c < best[0]:
                best = (c, J, B, tc)
print(f"  粗搜最优: 代价={best[0]:.4f}  J={best[1]}  B={best[2]}  tau_c={best[3]}")

c, J, B, tc = best
for step in (0.01, 0.003):
    improved = True
    while improved:
        improved = False
        for cand in ((J+step,B,tc),(J-step,B,tc),(J,B+step,tc),(J,B-step,tc),
                     (J,B,tc+step),(J,B,tc-step)):
            if cand[0] <= 0.05 or cand[1] <= 0.05 or cand[2] < 0:
                continue
            cc = cost(*cand, train)
            if cc < c - 1e-5:
                c, (J, B, tc), improved = cc, cand, True
print(f"  细化后:   代价={c:.4f}  J={J:.4f}  B={B:.4f}  tau_c={tc:.4f}")
print(f"  机械时间常数 J/B = {J/B:.3f} s    转折频率 = {B/(2*math.pi*J):.3f} Hz")

print("\n" + "="*76)
print("统一模型 vs 已有两组参数（自由仿真，训练/验证/遥控数据）")
print("="*76)
models = {
    "统一模型(本次)": (J, B, tc),
    "09-26 原标定":   (0.1503, 0.6605, 0.201),
    "09-29 新拟合":   (0.1504, 0.1250, 0.5837),
}
groups = [("训练集", train), ("验证集", val), ("★自然遥控", rc)]
print(f"  {'模型':<16s}" + "".join(f"{g:>14s}" for g, _ in groups))
for name, (jj, bb, tt) in models.items():
    row = f"  {name:<16s}"
    for g, files in groups:
        row += f"{cost(jj, bb, tt, files):14.3f}"
    print(row)
print("  （数值为归一化 RMS 误差：0=完美，1≈与预测均值同水平）")

print("\n" + "="*76)
print("★ 最佳模型对自然遥控数据的逐条表现")
print("="*76)
for p in rc:
    tau, w = cache[p]
    sim = simulate(tau, w[0], J, B, tc, DT_S)
    rms = np.sqrt(np.mean((sim-w)**2))
    print(f"\n  {os.path.basename(p)}")
    print(f"    时长 {len(w)*DT_S:.1f}s  |w|峰 {np.abs(w).max():.2f}  |tau|峰 {np.abs(tau).max():.2f} N*m")
    print(f"    RMS = {rms:.4f} rad/s   相关 = {np.corrcoef(w,sim)[0,1]:.4f}   "
          f"实测std {np.std(w):.3f} vs 仿真std {np.std(sim):.3f}")
