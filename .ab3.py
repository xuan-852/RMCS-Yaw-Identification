"""三组控制器 A/B 分析（A=kp_a10, B=kp_a25, C=kp_a25+FF），每组 2 次"""
import csv
import glob
import math
import os

import numpy as np

D = "/workspaces/RMCS/data/sessionE"
LIMIT = 2.5
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
    return {k: rows[:, n[k]] for k in
            ("condition_elapsed_s", "reference_angle", "measured_angle",
             "excitation_torque", "measured_velocity_imu", "pitch_world_angle",
             "pitch_temperature")}


def unwrap(a):
    o = np.zeros(len(a))
    for i in range(1, len(a)):
        o[i] = o[i-1] + np.remainder(a[i]-a[i-1]+np.pi, 2*np.pi) - np.pi
    return o


def metrics(p):
    d = load(p)
    t = d["condition_elapsed_s"]
    m = np.isfinite(d["reference_angle"])
    t = t[m]
    ref = np.degrees(unwrap(d["reference_angle"][m]))
    ang = np.degrees(unwrap(d["measured_angle"][m]))
    tau = d["excitation_torque"][m]
    pw = np.degrees(d["pitch_world_angle"][m])
    pt = d["pitch_temperature"][m]
    ref = ref - ref[0]; ang = ang - ang[0]
    e = ref - ang

    r = {"rms": float(np.sqrt(np.mean(e**2))), "max": float(np.max(np.abs(e)))}
    s = t >= SINE_START
    r["rms_sine"] = float(np.sqrt(np.mean(e[s]**2)))
    ov, st = [], []
    for k in range(4):
        a, b = int(k*HOLD/0.001), int((k+1)*HOLD/0.001)
        a, b = min(a, len(t)-1), min(b, len(t)-1)
        seg_e, seg_a = e[a:b], ang[a:b]
        tgt = ref[b-1]
        if abs(tgt) > 1e-3:
            arr = np.where(np.abs(seg_e) <= 2.0)[0]
            if len(arr):
                dr = 1.0 if tgt > ref[a] else -1.0
                past = (seg_a[arr[0]:] - tgt) * dr
                ov.append(float(max(0.0, past.max())))
        oi = np.where(np.abs(seg_e) >= 2.0)[0]
        st.append(float(oi[-1]) if len(oi) else 0.0)
    r["overshoot"] = float(np.mean(ov)) if ov else 0.0
    r["settle_ms"] = float(np.mean(st)) if st else 0.0
    r["tau_peak"] = float(np.max(np.abs(tau)))
    r["tau_pct"] = 100*float(np.max(np.abs(tau)))/LIMIT
    r["tau_rms"] = float(np.sqrt(np.mean(tau**2)))
    r["pitch_dev"] = float(pw.max()-pw.min())
    r["temp0"] = float(pt[0]); r["temp1"] = float(pt[-1])
    return r


groups = {}
for cid in range(21, 27):
    fs = glob.glob(f"{D}/yaw_c{cid}_track*.csv")
    if not fs:
        continue
    grp = {21: "A", 22: "B", 23: "C", 24: "A", 25: "B", 26: "C"}[cid]
    groups.setdefault(grp, []).append(metrics(fs[0]))

print("=" * 84)
print("三组控制器 A/B 实测（同一 ±30° 轨迹，过渡 0.6s，限幅 2.5 N*m）")
print("=" * 84)
print(f"\n  {'指标':<20s}{'A 原PID(kp_a=10)':>20s}{'B 整定(kp_a=25)':>18s}{'C 整定+前馈':>16s}")
print("  " + "-" * 78)
rows = [("rms", "跟踪RMS (deg)"), ("max", "峰值误差 (deg)"), ("rms_sine", "正弦段RMS (deg)"),
        ("overshoot", "阶跃超调 (deg)"), ("settle_ms", "调稳时间 (ms)"),
        ("tau_peak", "力矩峰值 (N*m)"), ("tau_pct", "占限幅 (%)"),
        ("pitch_dev", "pitch 波动 (deg)"), ("temp1", "末温 (°C)")]
for k, name in rows:
    line = f"  {name:<20s}"
    for g in ("A", "B", "C"):
        v = [x[k] for x in groups.get(g, [])]
        line += f"{np.mean(v):>12.3f}±{np.std(v):<5.3f}" if v else f"{'-':>20s}"
    print(line)

print("\n  === 相对 A 的改善 ===")
for g in ("B", "C"):
    if not groups.get(g) or not groups.get("A"):
        continue
    print(f"  {g}:")
    for k, name in (("rms", "跟踪RMS"), ("max", "峰值误差"), ("overshoot", "超调"),
                    ("settle_ms", "调稳"), ("tau_rms", "力矩RMS")):
        a = np.mean([x[k] for x in groups["A"]])
        b = np.mean([x[k] for x in groups[g]])
        d = (b-a)/a*100 if a else 0
        print(f"      {name}: {a:.3f} → {b:.3f}  ({d:+.1f}%)")
print(f"\n  ★ 饱和检查: 力矩峰值最高 {max(np.mean([x['tau_peak'] for x in groups[g]]) for g in groups):.3f} N*m"
      f" = {max(np.mean([x['tau_pct'] for x in groups[g]]) for g in groups):.0f}% 限幅"
      f"  → {'★ 未饱和 ✓' if max(np.mean([x['tau_pct'] for x in groups[g]]) for g in groups) < 95 else '✗ 仍饱和'}")
