"""Bieu do 3D cua ctrl3 (tu dong): moi cham = 1 run/seed trong khong gian (tre, mat goi, thong luong); cham lon = trung binh.
Doi chung: lambda co dinh 0,2/0,5/0,8 (duong bien lambda) va bo bao ve nguong co dinh. Chay thang file (khong can %run); tu tim so lieu.
Chi hien thi (plt.show); dat bien moi truong SAVE_DIR neu muon luu PNG."""
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def _find_dir():
    for d in (os.environ.get("LUANAN_DATA_DIR"), "/content/drive/MyDrive/Colab/#sualuanan/data1", "."):
        if d and os.path.isdir(d): return d
D = _find_dir()
P = lambda *a: os.path.join(D, *a)

CFG = {
    "ns3": dict(title="NS-3 (10 Mbps)", seeds=[2, 3, 4, 5, 6, 7, 8, 9, 11], rate=10.0, pref=P("results", "pref2"),
                shield=P("results", "dn_dcod", "da_tai_shield10.csv"), shield_lab="Chỉ bộ bảo vệ 10 ms"),
    "ukm": dict(title="UKM (2 Mbps)", seeds=[4, 5, 6, 7, 8, 9, 10], rate=None, pref=P("results_ukm", "pref_ukm2"),
                shield=P("ukm_shield50.csv"), shield_lab="Chỉ bộ bảo vệ 50 ms"),
}
STYLE = [("pref_fix20", "λ cố định 0,2", "#7f8c8d", "s"), ("pref_fix50", "λ cố định 0,5", "#2e86c1", "o"),
         ("pref_fix80", "λ cố định 0,8", "#8e44ad", "D"), ("keep_shield", None, "#e67e22", "^"),
         ("ctrl3", "ctrl3 tự động", "#c0392b", "*")]


def load(key):
    c = CFG[key]; parts = []
    f = os.path.join(c["pref"], "pref_metrics.csv")
    if os.path.exists(f):
        r = pd.read_csv(f); r = r[r.aqm.isin(["pref_fix20", "pref_fix50", "pref_fix80"])]
        if c["rate"] and "rate" in r: r = r[r.rate == c["rate"]]
        parts.append(r)
    f = os.path.join(c["pref"], "pref_metrics_ctrl3.csv")
    if os.path.exists(f):
        r = pd.read_csv(f); r = r[r.rate == c["rate"]] if c["rate"] and "rate" in r else r
        r = r.copy(); r["aqm"] = "ctrl3"; parts.append(r)
    if os.path.exists(c["shield"]):
        r = pd.read_csv(c["shield"]); r = r[r.aqm == "keep_shield"]
        if c["rate"] and "rate" in r: r = r[r.rate == c["rate"]]
        parts.append(r)
    else:
        print(f"[{key}] khong thay {c['shield']} -> bo cham bo bao ve")
    if not parts: raise SystemExit(f"[{key}] khong thay so lieu trong {D}")
    r = pd.concat(parts, ignore_index=True)
    return r[(r.cls == "All") & r.seed.isin(c["seeds"])]


def draw(key):
    c = CFG[key]; r = load(key)
    fig = plt.figure(figsize=(9.6, 8.2)); ax = fig.add_axes([.0, .15, 1, .81], projection="3d")
    for aqm, lab, col, mk in STYLE:
        g = r[r.aqm == aqm]
        if g.empty: continue
        lab = lab or c["shield_lab"]
        X, Y, Z = g.delay_ms.values, 100 * g.loss.values, g.thr_mbps.values
        ax.scatter(X, Y, Z, color=col, marker=mk, s=22, alpha=.35, depthshade=False)
        ax.scatter([X.mean()], [Y.mean()], [Z.mean()], color=col, marker=mk, s=330 if mk == "*" else 150, edgecolor="k", linewidth=1,
                   label=f"{lab} (n={len(g)})", depthshade=False, zorder=10)
    ax.set_xlabel("Trễ trung bình (ms)", labelpad=10); ax.set_ylabel("Mất gói (%)", labelpad=10); ax.set_zlabel("Thông lượng (Mbps)", labelpad=8)
    ax.view_init(24, -58); ax.set_box_aspect((1, 1, .8), zoom=1.12)
    ax.set_title(f"{c['title']}: ctrl3 (tự động) trong không gian trễ – mất gói – thông lượng", fontsize=12, fontweight="bold", pad=4)
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .045), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .012, "Chấm mờ = từng run/seed; chấm lớn = trung bình. Tốt hơn = trễ thấp, mất gói thấp, thông lượng cao.", ha="center", fontsize=9, color="#666")
    return fig


for k in ("ns3", "ukm"):
    f = draw(k)
    if os.environ.get("SAVE_DIR"): f.savefig(os.path.join(os.environ["SAVE_DIR"], f"ctrl3_3d_{k}.png"), dpi=110)
if not os.environ.get("SAVE_DIR"): plt.show()
