"""Fit a delayed first-order yaw model from the open-loop excitation phases.

Continuous model: J*w_dot + B*w + tau_c*sign(w) = tau_measured(t - delay) + bias
Exact zero-order-hold discretization (sign(w) held over each sample):
    w[k+1] = a*w[k] + b*tau_measured[k-d] + c*sign(w[k]) + c0
where a=exp(-B*dt/J), b=(1-a)/B, c=-b*tau_c.
"""
import csv
import math
import sys

import numpy as np


def load(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = {name: i for i, name in enumerate(header)}
        required = ("elapsed_s", "phase", "excitation_torque", "measured_torque", "measured_velocity_imu")
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = []
        for line_no, row in enumerate(reader, 2):
            try:
                rows.append([float(row[names[name]]) for name in required])
            except (ValueError, IndexError):
                raise ValueError(f"{path}:{line_no}: invalid required sample") from None
    data = np.asarray(rows, dtype=float)
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError(f"{path}: insufficient or non-finite required samples")
    t, phase, tau_command, tau_measured, w = data.T
    dt_values = np.diff(t)
    dt = float(np.median(dt_values))
    if dt <= 0 or not np.allclose(dt_values, dt, rtol=0.02, atol=1e-6):
        raise ValueError(f"{path}: timestamps are not uniformly sampled; resample before fitting")
    return {"t": t, "phase": phase, "tau_command": tau_command,
            "tau_measured": tau_measured, "w": w, "dt": dt}


def fit(data, delay_samples, torque_column="tau_measured", deadband=0.02):
    # Fit only open-loop sine and step excitation; exclude static and closed-loop track.
    selected = np.isin(data["phase"], (1, 2))
    d = int(delay_samples)
    boundaries = np.flatnonzero(np.diff(np.r_[False, selected, False]))
    blocks = list(zip(boundaries[::2], boundaries[1::2]))
    feature_blocks, target_blocks = [], []
    for start, end in blocks:
        t, tau, w = (data[key][start:end] for key in ("t", torque_column, "w"))
        if len(w) <= d + 4:
            continue
        if not np.allclose(np.diff(t), data["dt"], rtol=0.02, atol=1e-6):
            continue
        n_block = len(w) - d - 1
        sign_w = np.where(np.abs(w[d:d + n_block]) > deadband, np.sign(w[d:d + n_block]), 0.0)
        feature_blocks.append(np.column_stack((w[d:d + n_block], tau[:n_block], sign_w, np.ones(n_block))))
        target_blocks.append(w[d + 1:d + n_block + 1])
    if not feature_blocks:
        raise ValueError("delay leaves too few samples")
    X, y = np.vstack(feature_blocks), np.concatenate(target_blocks)
    n = len(y)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ theta
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return theta, r2, n


def params_from(theta, dt):
    a, b, c, c0 = theta
    if not (0.0 < a < 1.0 and b > 0.0):
        return float("nan"), float("nan"), float("nan"), c0 / b if b else float("nan")
    B = (1.0 - a) / b
    J = -B * dt / math.log(a)
    tau_c = -c / b
    bias = c0 / b
    return J, B, tau_c, bias


def validate(theta, data, delay_samples, torque_column="tau_measured", deadband=0.02):
    selected = np.isin(data["phase"], (1, 2))
    d = int(delay_samples)
    boundaries = np.flatnonzero(np.diff(np.r_[False, selected, False]))
    feature_blocks, target_blocks = [], []
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        t, tau, w = (data[key][start:end] for key in ("t", torque_column, "w"))
        if len(w) <= d + 4 or not np.allclose(np.diff(t), data["dt"], rtol=0.02, atol=1e-6):
            continue
        n_block = len(w) - d - 1
        sign_w = np.where(np.abs(w[d:d + n_block]) > deadband, np.sign(w[d:d + n_block]), 0.0)
        feature_blocks.append(np.column_stack((w[d:d + n_block], tau[:n_block], sign_w, np.ones(n_block))))
        target_blocks.append(w[d + 1:d + n_block + 1])
    if not feature_blocks:
        raise ValueError("validation has no usable open-loop blocks")
    X, y = np.vstack(feature_blocks), np.concatenate(target_blocks)
    pred = X @ theta
    return 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} TRAIN.csv [VALIDATION.csv]")
    train = load(sys.argv[1])
    dt = train["dt"]
    max_delay_samples = int(round(0.030 / dt))
    candidates = []
    for delay in range(max_delay_samples + 1):
        theta, r2, n = fit(train, delay)
        if np.isfinite(r2):
            candidates.append((r2, delay, theta, n))
    if not candidates:
        raise SystemExit("no valid delay candidate")
    r2, delay, theta, n = max(candidates, key=lambda item: item[0])
    J, B, tau_c, bias = params_from(theta, dt)
    print(f"train: {sys.argv[1]}  input=measured_torque  open-loop rows={n}  dt={dt * 1000:.3f} ms")
    print(f"best delay={delay * dt * 1000:.2f} ms  fit R2={r2:.5f}")
    print("discrete: w[k+1] = " + " + ".join(f"{v:.6g}*{name}" for v, name in zip(theta, ("w[k]", "tau_measured[k-d]", "sign(w[k])", "1"))))
    print(f"J={J:.5f} kg*m^2  B={B:.5f} N*m*s/rad  tau_c={tau_c:.5f} N*m  torque_bias={bias:.5f} N*m")
    if not np.isfinite([J, B, tau_c]).all() or J <= 0 or B <= 0 or tau_c < 0:
        print("WARNING: fitted coefficients do not map to a physically valid J>0, B>0, tau_c>=0 model")
    if len(sys.argv) > 2:
        val = load(sys.argv[2])
        if not math.isclose(val["dt"], dt, rel_tol=0.02, abs_tol=1e-6):
            raise SystemExit("training and validation sample periods differ")
        print(f"validate on {sys.argv[2]}: open-loop R2={validate(theta, val, delay):.5f}")
