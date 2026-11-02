# -*- coding: utf-8 -*-
"""Ve lai 2 bieu do so sanh 3 co che offline (NS-3 va UKM) dang LINE: MOI CO CHE = 1 DUONG qua cac lan chay; duong dut = trung binh, dai mo = +-1 do lech chuan.
Doc truc tiep aqm_wide_<chi so><hau to>.csv (hang = seed/run, cot = co che). Chay thang file (khong can %run), CHI HIEN THI; dat SAVE_DIR neu muon luu PNG.
NS-3: hau to _offline ; UKM: hau to _ukm."""
import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

D = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
SEARCH = [D, D + "/data2", "."]
ORDER = ["dcodel", "dn_dcod_khong_nhan", "dn_dcod"]
LBL = {"dcodel": "dCoDel", "dn_dcod_khong_nhan": "DN-dCoD\n(không nhãn)", "dn_dcod": "DN-dCoD\n(nhúng nhãn)"}
COL = {"dcodel": "#1f3fff", "dn_dcod_khong_nhan": "#f39c12", "dn_dcod": "#e0201b"}
MK = {"dcodel": "o", "dn_dcod_khong_nhan": "s", "dn_dcod": "D"}
SUF = {"NS-3": "_offline", "UKM": "_ukm"}
SPEC = {"NS-3": [("avg_delay_ms", "Độ trễ TB (ms)", 1, "{:.2f}"), ("loss_rate", "Tỷ lệ mất gói (%)", 100, "{:.2f}"),
                 ("throughput_mbps", "Thông lượng (Mbps)", 1, "{:.3f}"), ("jain_index", "Chỉ số Jain", 1, "{:.4f}")],
        "UKM": [("avg_delay_ms", "Độ trễ TB (ms)", 1, "{:.1f}"), ("p95_delay_ms", "Độ trễ p95 (ms)", 1, "{:.0f}"),
                ("loss_rate", "Tỷ lệ mất gói (%)", 100, "{:.1f}"), ("throughput_mbps", "Thông lượng (Mbps)", 1, "{:.3f}")]}
TITLE = {"NS-3": "NS-3 (10 Mbps)", "UKM": "UKM (2 Mbps)"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})


LOP = {"avg_delay_ms": "delay_ms", "p95_delay_ms": "p95_delay_ms", "loss_rate": "loss", "throughput_mbps": "thr_mbps"}


def load_wide(dom, m):
    """Bang rong (hang = lan chay, cot = co che). Uu tien aqm_wide_<chi so><hau to>.csv; neu khong co (vd. p95 cua UKM)
    thi lay tu ukm_lop.csv (phan tich theo lop, dong cls = All) cua ukm_mo_rong.py --stage lop."""
    for b in SEARCH:
        f = os.path.join(b, f"aqm_wide_{m}{SUF[dom]}.csv")
        if os.path.exists(f): return pd.read_csv(f, index_col=0)[ORDER]
    if dom == "UKM" and m in LOP:
        for b in SEARCH:
            f = os.path.join(b, "ukm_lop.csv")
            if os.path.exists(f):
                r = pd.read_csv(f)
                if LOP[m] in r:
                    r = r[(r.cls == "All") & r.aqm.isin(ORDER)]
                    print(f"[UKM] {m}: lay tu {f}"); return r.pivot(index="seed", columns="aqm", values=LOP[m])[ORDER]
    return None


def draw(dom):
    """Moi co che = 1 duong (gia tri qua tung lan chay); duong dut = trung binh, dai mo = +-1 do lech chuan."""
    LB1 = {k: v.replace("\n", " ") for k, v in LBL.items()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11)); n = None
    for ax, (m, title, f, fmt) in zip(axes.ravel(), SPEC[dom]):
        df = load_wide(dom, m)
        if df is None:
            ax.set_title(title + " (thiếu file)"); ax.axis("off"); print(f"[{dom}] thieu aqm_wide_{m}{SUF[dom]}.csv (va ukm_lop.csv)"); continue
        df = df.dropna() * f; n = len(df); x = np.arange(1, n + 1)
        for k in ORDER:
            y = df[k].values; mu, sd = y.mean(), y.std(ddof=1)
            ax.fill_between([x[0], x[-1]], mu - sd, mu + sd, color=COL[k], alpha=.13, lw=0)
            ax.axhline(mu, color=COL[k], ls="--", lw=1.3, alpha=.9)
            ax.plot(x, y, color=COL[k], lw=2.2, marker=MK[k], ms=7, mec="white", mew=1.2, label=f"{LB1[k]}: " + fmt.format(mu) + " ± " + fmt.format(sd))
        ax.set_title(title); ax.set_xlabel("Lần chạy (seed)"); ax.set_xticks(x)
        ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False)
    fig.suptitle(f"{TITLE[dom]}: so sánh 3 cơ chế offline qua {n or '?'} lần chạy (đường nét đứt = trung bình; dải mờ = ± độ lệch chuẩn)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); return fig


for d in ("NS-3", "UKM"):
    f = draw(d)
    if os.environ.get("SAVE_DIR"): f.savefig(os.path.join(os.environ["SAVE_DIR"], f"line_{d}.png"), dpi=130)
if not os.environ.get("SAVE_DIR"): plt.show()
