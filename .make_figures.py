import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "/workspaces/RMCS/data"
OUT = "/workspaces/RMCS/data/figures"
os.makedirs(OUT, exist_ok=True)

A, B = 0.99560, 6.655e-03   # discrete fit (train = set 2)
C = -1.34e-03
DT = 0.001
J = DT / B
TAU_C = abs(C) * J / DT


def load(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        names = {n: i for i, n in enumerate(header)}
        parsed = ([float(x) if x not in ("", "nan", "-nan") else math.nan for x in row]
                  for row in reader)
        rows = np.array([r + [math.nan] * (len(header) - len(r)) for r in parsed])
    g = lambda n: rows[:, names[n]]
    return {n: g(n) for n in header}


def unwrap_series(angle):
    out = np.zeros(len(angle))
    for i in range(1, len(angle)):
        out[i] = out[i - 1] + np.remainder(angle[i] - angle[i - 1] + np.pi, 2 * np.pi) - np.pi
    return out


set2 = load(f"{DATA}/yaw_identification_2026-09-26_09-52-07.csv")
set1 = load(f"{DATA}/yaw_identification_2026-09-26_09-47-20.csv")
base = load(f"{DATA}/yaw_baseline_2026-09-26.csv")
model = load(f"{DATA}/yaw_model_2026-09-26.csv")

# ---- fig 1: protocol overview (set 2) ----
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
t = set2["elapsed_s"]
ax[0].plot(t, set2["excitation_torque"], lw=0.5, color="tab:red")
ax[0].set_ylabel("torque (N·m)")
ax[0].set_title("Excitation protocol overview (sine amplitude 1.0 N·m)")
ax[1].plot(t, set2["measured_velocity_imu"], lw=0.5, color="tab:blue")
ax[1].set_ylabel("yaw velocity (rad/s)")
ax[2].plot(t, set2["measured_angle"], lw=0.5, color="tab:green")
ax[2].set_ylabel("yaw angle (rad)")
ax[2].set_xlabel("time (s)")
for a in ax:
    for x0, x1, c, lab in ((0, 15, "#eeeeee", "static"), (15, 45, "#e3f0ff", "sine"),
                           (45, 50, "#ffe9e3", "steps")):
        a.axvspan(x0, x1, color=c, zorder=0)
ax[0].text(7.5, 0.8, "static", ha="center")
ax[0].text(30, 0.8, "sine 1 Hz", ha="center")
ax[0].text(47.5, 0.8, "steps", ha="center")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_protocol_overview.png", dpi=150)
plt.close()

# ---- fig 2: sine zoom ----
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
m = (set2["elapsed_s"] >= 15) & (set2["elapsed_s"] <= 30)
ax[0].plot(set2["elapsed_s"][m], set2["excitation_torque"][m], lw=0.8, label="excitation torque", color="tab:red")
ax[0].plot(set2["elapsed_s"][m], set2["measured_torque"][m], lw=0.5, alpha=0.6, label="measured torque")
ax[0].set_ylabel("torque (N·m)")
ax[0].legend(loc="upper right", fontsize=8)
ax[1].plot(set2["elapsed_s"][m], set2["measured_velocity_imu"][m], lw=0.8, color="tab:blue")
ax[1].set_ylabel("yaw velocity (rad/s)")
ax[1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_sine_response.png", dpi=150)
plt.close()

# ---- fig 3: model free-run simulation vs measured (set 2, motion phase) ----
m = set2["elapsed_s"] >= 15
tau = set2["excitation_torque"][m]
w_meas = set2["measured_velocity_imu"][m]
w_sim = np.zeros(len(tau))
w_sim[0] = w_meas[0]
for k in range(1, len(tau)):
    s = 1.0 if w_sim[k - 1] >= 0 else -1.0
    w_sim[k] = A * w_sim[k - 1] + B * tau[k - 1] + C * s
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(set2["elapsed_s"][m], w_meas, lw=0.6, alpha=0.75, label="measured")
ax.plot(set2["elapsed_s"][m], w_sim, lw=0.6, alpha=0.85,
        label=f"model free-run (J={J:.3f}, B={0.66:.2f}, τc={TAU_C:.2f})")
ax.set_xlabel("time (s)")
ax.set_ylabel("yaw velocity (rad/s)")
ax.set_title("Free-run model simulation vs measured (35 s, no feedback from measurement)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_free_run.png", dpi=150)
plt.close()

# ---- fig 4: A/B tracking ----
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
tb = base["elapsed_s"][base["phase"] == 3] - 50.0
tm = model["elapsed_s"][model["phase"] == 3] - 50.0
ref = np.radians(30) * 0  # reference in degrees relative
rb = base["reference_angle"][base["phase"] == 3]
rm = model["reference_angle"][model["phase"] == 3]
rb = np.degrees(rb - rb[0])
rm = np.degrees(rm - rm[0])
angb = np.degrees(unwrap_series(base["measured_angle"][base["phase"] == 3]))
angm = np.degrees(unwrap_series(model["measured_angle"][model["phase"] == 3]))
angb = angb - angb[0]
angm = angm - angm[0]
ax[0].plot(tb, rb, "k--", lw=1, label="reference")
ax[0].plot(tb, angb, lw=0.8, alpha=0.8, label="baseline (original ctrl)")
ax[0].plot(tm, angm, lw=0.8, alpha=0.8, label="model (feedforward)")
ax[0].set_ylabel("yaw angle (deg, relative)")
ax[0].legend(fontsize=8, loc="lower right")
ax[0].set_title("Tracking phase A/B comparison (steps ±30° then 0.5 Hz sine)")
errb = rb - angb
errm = rm - angm
ax[1].plot(tb, errb, lw=0.8, label=f"baseline error (rms={np.sqrt(np.mean(errb**2)):.2f}°)")
ax[1].plot(tm, errm, lw=0.8, label=f"model error (rms={np.sqrt(np.mean(errm**2)):.2f}°)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_ylabel("tracking error (deg)")
ax[1].set_xlabel("time (s)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_ab_tracking.png", dpi=150)
plt.close()

# ---- fig 5: first step zoom ----
fig, ax = plt.subplots(figsize=(10, 4))
m1 = (tb >= 0) & (tb <= 2.5)
ax.plot(tb[m1], rb[m1], "k--", lw=1, label="reference")
ax.plot(tb[m1], angb[m1], lw=1.2, label="baseline")
ax.plot(tm[m1], angm[m1], lw=1.2, label="model")
ax.set_xlabel("time (s)")
ax.set_ylabel("yaw angle (deg)")
ax.set_title("First step response detail (+30°)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_step_zoom.png", dpi=150)
plt.close()

print("figures written:", sorted(os.listdir(OUT)))
