import io
P = "/workspaces/RMCS/.predict_closed_loop.py"
s = io.open(P, encoding="utf-8").read()
old_start = s.index("def metrics(ref, ang, tau, transition):")
old_end = s.index("print(\"=\" * 78)")
new = '''def metrics(ref, ang, tau, transition):
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


'''
s = s[:old_start] + new + s[old_end:]
io.open(P, "w", encoding="utf-8").write(s)
print("metrics fixed")
