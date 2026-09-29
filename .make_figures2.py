"""报告图表生成（09-29 更新版）"""
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = "/workspaces/RMCS/data/figures"
os.makedirs(OUT, exist_ok=True)
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
    return {
        "t": rows[:, n[tk]],
        "tau": rows[:, n["measured_torque"]],
        "exc": rows[:, n["excitation_torque"]],
        "w": rows[:, n["measured_velocity_imu"]],
        "ref": rows[:, n["reference_angle"]],
        "ang": rows[:, n["measured_angle"]],
    }


def simulate(tau, w0, J, B, tc):
    w = np.empty(len(tau)); w[0] = w0
    for i in range(len(tau)-1):
        s = 1.0 if w[i] > 0 else (-1.0 if w[i] < 0 else 0.0)
        w[i+1] = w[i] + DT*(tau[i] - B*w[i] - tc*s)/J
    return w


def unwrap(a):
    o = np.zeros(len(a))
    for i in range(1, len(a)):
        o[i] = o[i-1] + np.remainder(a[i]-a[i-1]+np.pi, 2*np.pi) - np.pi
    return o


# ---------- fig6: 统一模型 vs 两批数据 ----------
models = {"Unified\n(J=.137 B=.530 tc=.360)": (0.137, 0.530, 0.360),
          "09-26 fit\n(J=.150 B=.661 tc=.201)": (0.1503, 0.6605, 0.201),
          "09-29-only fit\n(J=.150 B=.125 tc=.584)": (0.1504, 0.1250, 0.5837)}
err_table = {
    "Unified\n(J=.137 B=.530 tc=.360)": (0.333, 0.162, 1.471),
    "09-26 fit\n(J=.150 B=.661 tc=.201)": (0.883, 0.140, 1.553),
    "09-29-only fit\n(J=.150 B=.125 tc=.584)": (0.486, 0.120, 2.678),
}
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = ["Train (both datasets)", "Holdout (steps+track)", "RC random control"]
x = np.arange(len(labels)); wbar = 0.26
for i, (name, vals) in enumerate(err_table.items()):
    ax.bar(x + (i-1)*wbar, vals, wbar, label=name)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Normalized RMS error (lower=better)")
ax.set_title("Unified model vs single-dataset models (free-run simulation error)")
ax.legend(fontsize=7.5); ax.grid(axis="y", ls="--", alpha=0.3)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_model_compare.png", dpi=150); plt.close()

# ---------- fig7: 遥控数据 实测 vs 仿真 ----------
J, B, TC = 0.137, 0.530, 0.360
rc = sorted(glob.glob("/workspaces/RMCS/data/sessionC/yaw_c27_rc*.csv"))[0]
d = load(rc); t = d["t"]
sim = simulate(d["tau"], d["w"][0], J, B, TC)
fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
ax[0].plot(t, d["tau"], lw=0.5, color="tab:red")
ax[0].set_ylabel("applied torque (N$\\cdot$m)")
ax[0].set_title("RC random control: measured vs unified-model free-run")
ax[0].grid(ls="--", alpha=0.3)
ax[1].plot(t, d["w"], lw=0.6, alpha=0.8, label="measured")
ax[1].plot(t, sim, lw=0.6, alpha=0.85, label=f"model free-run (r={np.corrcoef(d['w'],sim)[0,1]:.3f})")
ax[1].set_ylabel("yaw velocity (rad/s)"); ax[1].set_xlabel("time (s)")
ax[1].legend(fontsize=8); ax[1].grid(ls="--", alpha=0.3)
for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_rc_fit.png", dpi=150); plt.close()

# ---------- fig8: A/B 对照（含限幅饱和证据）----------
def grab(pat):
    return {os.path.basename(p).split("_")[1]: p for p in glob.glob(f"/workspaces/RMCS/data/sessionC/{pat}")}
base_files = [glob.glob(f"/workspaces/RMCS/data/sessionC/yaw_c{n}_track*.csv")[0] for n in (21, 23, 25)]
model_files = [glob.glob(f"/workspaces/RMCS/data/sessionC/yaw_c{n}_track*.csv")[0] for n in (22, 24, 26)]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# 左：跟踪误差
for tag, files, col in (("baseline", base_files, "tab:red"), ("model (FF)", model_files, "tab:blue")):
    errs = []
    for p in files:
        d = load(p)
        m = np.isfinite(d["ref"])
        r = np.degrees(unwrap(d["ref"][m])); a = np.degrees(unwrap(d["ang"][m]))
        errs.append(r - r[0] - (a - a[0]))
    e = np.mean(errs, axis=0)
    ax[0].plot(np.arange(len(e))*DT, e, lw=0.8, color=col, label=tag)
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("tracking error (deg)")
ax[0].set_title("A/B tracking error (mean of 3 runs each)")
ax[0].legend(fontsize=8); ax[0].grid(ls="--", alpha=0.3)
# 右：力矩饱和证据
tau_b = [np.abs(load(p)["exc"]).max() for p in base_files]
tau_m = [np.abs(load(p)["exc"]).max() for p in model_files]
ax[1].bar(["baseline", "model"], [np.mean(tau_b), np.mean(tau_m)],
          yerr=[np.std(tau_b), np.std(tau_m)], color=["tab:red", "tab:blue"], width=0.5)
ax[1].axhline(2.5, color="k", ls="--", lw=1.2, label="torque limit 2.5 N$\\cdot$m")
ax[1].set_ylabel("peak |command torque| (N$\\cdot$m)")
ax[1].set_title("Both hit the torque cap -> FF advantage masked")
ax[1].legend(fontsize=8); ax[1].grid(axis="y", ls="--", alpha=0.3)
for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_ab_2026-09-29.png", dpi=150); plt.close()

print("figures written:", sorted(os.listdir(OUT)))
