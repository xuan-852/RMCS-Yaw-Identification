import io
P = "/workspaces/RMCS/.ab_stats.py"
s = io.open(P, encoding="utf-8").read()

old = """        tgt = np.sign(ref[seg][-1]) * np.radians(30) if abs(ref[seg][-1]) > 1e-3 else 0.0
        if abs(tgt) > 1e-3:
            os_ = np.degrees(np.max(np.abs(e[np.sign(e) != np.sign(tgt)]))) if (np.sign(e) != np.sign(tgt)).any() else 0.0
            ov.append(os_)"""
new = """        tgt = ref[seg][-1]
        if abs(tgt) > 1e-3:
            # 正确超调定义：角度首次进入目标 ±2° 之后，越过目标的最大过冲量
            arrive = np.where(np.abs(e) <= np.radians(2.0))[0]
            if len(arrive):
                d = 1.0 if tgt > ref[seg][0] else -1.0
                past = (ang[seg][arrive[0]:] - tgt) * d
                ov.append(np.degrees(max(0.0, float(past.max()))))"""
assert old in s, "overshoot block not found"
s = s.replace(old, new, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("ab_stats overshoot metric fixed")
