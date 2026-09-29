"""离线闭环预测
1) 先用仿真器重现 09-29 的 baseline 实测（阶跃过渡 0.3s + 2.5 N*m 限幅）→ 验证仿真器可信
2) 再预测三种控制器在"模型算过可行性"的新轨迹（过渡 0.45s）上的表现

模型: J=0.137  B=0.530  tau_c=0.360   (统一模型)
控制器(原版语义): vel_ref = kp_a*e_ang ; tau = kp_v*(vel_ref - w) + ki_v*I,  I += (vel_ref - w)
"""
import csv
import math

import numpy as np

J, B, TAU_C = 0.137, 0.530, 0.360
DT = 0.001
LIMIT = 2.5
HALF = math.radians(30.0)


def reference(t, transition):
    """track 轨迹：4 拍阶跃(+30/-30/+30/0，各 2.5s) → 0.5Hz ±30° 正弦"""
    hold, tgts = 2.5, [1.0, -1.0, 1.0, 0.0]
    step_phase = 4 * hold
    if t < step_phase:
        k = int(t / hold)
        s0 = k * hold
        u = min(max((t - s0) / transition, 0.0), 1.0)
        u = u**3 * (u * (6 * u - 15) + 10)          # smootherstep
        prev = 0.0 if k == 0 else tgts[k - 1]
        return (prev + (tgts[k] - prev) * u) * HALF
    return HALF * math.sin(2 * math.pi * 0.5 * (t - step_phase))


def run(kp_a, kp_v, ki_v, use_ff, transition, tau_limit, dur):
    n = int(dur / DT) + 1
    ref = np.array([reference(i * DT, transition) for i in range(n)])
    ref_v = np.gradient(ref, DT)
    ang = np.zeros(n + 1); w = np.zeros(n + 1); tau = np.zeros(n)
    I = 0.0
    for i in range(1, n):
        e_ang = ref[i] - ang[i]
        vel_ref = kp_a * e_ang
        e_vel = vel_ref - w[i]
        I = min(max(I + e_vel, -5.0), 5.0)
        tq = kp_v * e_vel + ki_v * I
        if use_ff:
            ff = J * ref_v[i] * 0 + B * 0  # 占位
        if use_ff:
            a_ref = (ref_v[i] - ref_v[i - 1]) / DT
            tq += J * a_ref + B * ref_v[i] + TAU_C * (1.0 if ref_v[i] >= 0 else -1.0)
        tq = min(max(tq, -tau_limit), tau_limit)
        tau[i] = tq
        s = 1.0 if w[i] > 0 else (-1.0 if w[i] < 0 else 0.0)
        w[i + 1] = w[i] + DT * (tq - B * w[i] - TAU_C * s) / J
        ang[i + 1] = ang[i] + DT * w[i + 1]
    return ref, ang, w, tau


def metrics(ref, ang, tau, transition):
    """修正版指标：超调 = 越过目标后的最大过冲（不是初始瞬态）；调稳 = 最后一次 |e|>=2° 的时刻(ms)"""
    ang = ang[:len(ref)]
    e = ref - ang
    out = {"rms": np.degrees(np.sqrt(np.mean(e**2))),
           "max": np.degrees(np.max(np.abs(e))),
           "tau_peak": np.max(np.abs(tau)),
           "tau_pct": 100 * np.max(np.abs(tau)) / LIMIT}
    ov, st = [], []
    for k in range(4):
        s0 = int(k * 2.5 / DT); e0 = int((k + 1) * 2.5 / DT)
        seg_e = e[s0:e0]; seg_a = ang[s0:e0]
        tgt = ref[e0 - 1]
        prev = ref[s0 - 1] if s0 > 0 else 0.0
        direction = tgt - prev
        if abs(direction) > 1e-6:
            past = (seg_a - tgt) * (1.0 if direction > 0 else -1.0)
            ov.append(np.degrees(max(0.0, past.max())))
        # 调稳：最后一个 |e| >= 2° 的样本序号（1kHz → 单位 ms）
        oi = np.where(np.abs(np.degrees(seg_e)) >= 2.0)[0]
        st.append(oi[-1] if len(oi) else 0)
    out["overshoot"] = np.mean(ov) if ov else 0.0
    out["settle_ms"] = np.mean(st) if st else 0.0
    return out


print("=" * 78)
print("第 1 步：仿真器验证 —— 重现 09-29 实测的 baseline（过渡 0.3s + 2.5 N*m 限幅）")
print("=" * 78)
ref, ang, w, tau = run(10.0, 13.0, 0.02, False, 0.30, LIMIT, 20.0)
m = metrics(ref, ang, tau, 0.30)
print(f"  仿真: 跟踪RMS={m['rms']:.2f}°  峰值误差={m['max']:.1f}°  超调={m['overshoot']:.1f}°  "
      f"调稳={m['settle_ms']:.0f}ms  力矩峰值={m['tau_peak']:.2f}N·m ({m['tau_pct']:.0f}%)")
print(f"  实测: 跟踪RMS=5.46°  峰值误差=39.8°  超调=33.6°  调稳=418ms  力矩峰值=2.50N·m (100%)")
print("  → 若超调量级吻合，说明仿真器抓住了主要机理（限幅导致的减速段缺失）")

print("\n" + "=" * 78)
print("第 2 步：新轨迹可行性（过渡 0.45s）—— 同样的 baseline")
print("=" * 78)
ref, ang, w, tau = run(10.0, 13.0, 0.02, False, 0.45, LIMIT, 20.0)
m0 = metrics(ref, ang, tau, 0.45)
print(f"  baseline(kp_a=10): RMS={m0['rms']:.2f}°  超调={m0['overshoot']:.2f}°  "
      f"调稳={m0['settle_ms']:.0f}ms  力矩峰值={m0['tau_peak']:.2f}N·m ({m0['tau_pct']:.0f}%)")

print("\n" + "=" * 78)
print("第 3 步：三种控制器对比（新轨迹，过渡 0.45s）")
print("=" * 78)
variants = [
    ("A 原PID           (kp_a=10, 无FF)", 10.0, 13.0, 0.02, False),
    ("B 模型整定PID     (kp_a=25, 无FF)", 25.0, 13.0, 0.02, False),
    ("C 模型整定PID+前馈(kp_a=25, 有FF)", 25.0, 13.0, 0.02, True),
]
print(f"  {'方案':<38s}{'RMS°':>7s}{'峰值°':>7s}{'超调°':>7s}{'调稳ms':>8s}{'力矩峰':>8s}{'占限幅':>7s}")
print("  " + "-" * 74)
res = {}
for name, ka, kv, ki, ff in variants:
    ref, ang, w, tau = run(ka, kv, ki, ff, 0.45, LIMIT, 20.0)
    m = metrics(ref, ang, tau, 0.45)
    res[name] = m
    print(f"  {name:<38s}{m['rms']:7.2f}{m['max']:7.1f}{m['overshoot']:7.2f}"
          f"{m['settle_ms']:8.0f}{m['tau_peak']:8.2f}{m['tau_pct']:6.0f}%")

print("\n  === 相对 A 的改善 ===")
a = res[variants[0][0]]
for name, *_ in variants[1:]:
    b = res[name]
    print(f"  {name}")
    for k, lab in (("rms", "跟踪RMS"), ("overshoot", "超调"), ("settle_ms", "调稳")):
        d = (b[k] - a[k]) / a[k] * 100 if a[k] else 0
        print(f"      {lab}: {a[k]:.2f} → {b[k]:.2f}  ({d:+.1f}%)")
