"""A/B 对照统计：3 组 baseline vs 3 组 model（track 工况，±30° 脚本轨迹）"""
import csv
import glob
import math

import numpy as np

D = "/workspaces/RMCS/data/sessionC"
HOLD = 2.5
SINE_START = 10.0


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
        "ref": rows[:, n["reference_angle"]],
        "ang": rows[:, n["measured_angle"]],
        "tau": rows[:, n["excitation_torque"]],
        "w": rows[:, n["measured_velocity_imu"]],
    }


def unwrap(a):
    o = np.zeros(len(a))
    for i in range(1, len(a)):
        o[i] = o[i-1] + np.remainder(a[i]-a[i-1]+np.pi, 2*np.pi) - np.pi
    return o


def metrics(p):
    d = load(p)
    m = np.isfinite(d["ref"])
    t, ref = d["t"][m], np.degrees(unwrap(d["ref"][m]))
    ang = np.degrees(unwrap(d["ang"][m]))
    tau = d["tau"][m]
    ref = ref - ref[0]; ang = ang - ang[0]
    err = ref - ang

    res = {"rms_all": np.sqrt(np.mean(err**2)), "max_all": np.max(np.abs(err))}
    # 正弦段
    s = t >= SINE_START
    res["rms_sine"] = np.sqrt(np.mean(err[s]**2))
    # 阶跃段
    ov, st = [], []
    for k in range(4):
        a, b = k*HOLD, (k+1)*HOLD
        seg = (t >= a) & (t < b)
        if seg.sum() < 50:
            continue
        e, tt = err[seg], t[seg]
        tgt = ref[seg][-1]
        if abs(tgt) > 1e-3:
            # 正确超调定义：角度首次进入目标 ±2° 之后，越过目标的最大过冲量
            arrive = np.where(np.abs(e) <= np.radians(2.0))[0]
            if len(arrive):
                d = 1.0 if tgt > ref[seg][0] else -1.0
                past = (ang[seg][arrive[0]:] - tgt) * d
                ov.append(np.degrees(max(0.0, float(past.max()))))
        out = np.where(np.abs(e) >= 2.0)[0]
        st.append((tt[out[-1]] - a)*1000 if len(out) else 0.0)
    res["overshoot"] = np.mean(ov) if ov else 0.0
    res["settle_ms"] = np.mean(st) if st else 0.0
    res["tau_rms"] = np.sqrt(np.mean(tau**2))
    res["tau_peak"] = np.max(np.abs(tau))
    res["tau_smooth"] = np.std(np.diff(tau))*1e6/1e3
    return res


base = sorted(glob.glob(f"{D}/yaw_c2[1,3,5]*_track_*.csv")) or \
       sorted(glob.glob(f"{D}/yaw_c21_track*.csv")) + sorted(glob.glob(f"{D}/yaw_c23_track*.csv")) + \
       sorted(glob.glob(f"{D}/yaw_c25_track*.csv"))
model = sorted(glob.glob(f"{D}/yaw_c22_track*.csv")) + sorted(glob.glob(f"{D}/yaw_c24_track*.csv")) + \
        sorted(glob.glob(f"{D}/yaw_c26_track*.csv"))

print("=" * 76)
print("A/B 对照：baseline（原控制器） vs model（模型前馈）")
print("=" * 76)
print(f"\n  baseline 文件数: {len(base)},  model 文件数: {len(model)}")

rows_b = [metrics(p) for p in base]
rows_m = [metrics(p) for p in model]

keys = [("rms_all", "跟踪 RMS 误差 (deg)", False),
        ("max_all", "峰值误差 (deg)", False),
        ("rms_sine", "正弦段 RMS (deg)", False),
        ("overshoot", "阶跃超调 (deg)", False),
        ("settle_ms", "调稳时间 (ms)", False),
        ("tau_rms", "力矩 RMS (N*m)", False),
        ("tau_peak", "力矩峰值 (N*m)", False),
        ("tau_smooth", "力矩平滑度", False)]

print(f"\n  {'指标':<22s}{'baseline':>16s}{'model':>16s}{'改善':>12s}")
print("  " + "-" * 66)
for k, name, _ in keys:
    b = np.mean([r[k] for r in rows_b]); bs = np.std([r[k] for r in rows_b])
    m = np.mean([r[k] for r in rows_m]); ms = np.std([r[k] for r in rows_m])
    if b != 0:
        imp = f"{(m-b)/b*100:+.1f}%"
    else:
        imp = "-"
    print(f"  {name:<22s}{b:9.3f}±{bs:<6.3f}{m:9.3f}±{ms:<6.3f}{imp:>12s}")
