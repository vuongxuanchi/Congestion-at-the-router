# -*- coding: utf-8 -*-
"""Ve 3D tu KET QUA THAT cua sweep_ctrl3.py (bo tu dieu chinh TU DONG ctrl3) - CHI HIEN THI, KHONG LUU FILE.
  2x2 cho moi bo du lieu (NS-3, UKM):  [tre | mat goi] theo (cua so, buoc lambda)  va  [tre | mat goi] theo (he so nguong kappa, buoc lambda)
  (moi be mat la TRUNG BINH cua bien con lai; cham do = cau hinh da chot; sao xanh = cau hinh tre thap nhat)
Chay truc tiep (dan vao 1 cell Colab). Tu dong lay sweep_ctrl3_ns3.csv / sweep_ctrl3_ukm.csv (LUANAN_DATA_DIR hoac thu muc Drive).
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def _find_dir():
    for d in (os.environ.get("LUANAN_DATA_DIR"), "/content/drive/MyDrive/Colab/#sualuanan/data1", "."):
        if d and os.path.exists(os.path.join(d, "sweep_ctrl3_ukm.csv")) or d and os.path.exists(os.path.join(d, "sweep_ctrl3_ns3.csv")):
            return d
    return "."

D = _find_dir()
NAME = {"ns3": "NS-3 (10 Mbps)", "ukm": "UKM (2 Mbps)"}


def load(key):
    f = os.path.join(D, f"sweep_ctrl3_{key}.csv")
    if not os.path.exists(f):
        print(f"[{key}] khong thay {f} -> bo qua"); return None
    r = pd.read_csv(f)
    return r.groupby(["kap", "step", "win"]).agg(delay=("delay_ms", "mean"), loss=("loss", lambda s: 100 * s.mean()), n=("seed", "nunique")).reset_index()


def surf(ax, g, a, b, z, cm, title, zlabel, default, best, fmt):
    """be mat z theo (a, b); cac bien con lai lay trung binh. a,b la cot cua g."""
    A, B = sorted(g[a].unique()), sorted(g[b].unique())
    Z = np.array([[g[(g[a] == x) & (g[b] == y)][z].mean() for x in A] for y in B])      # hang = b, cot = a
    ia, ib = np.arange(len(A), dtype=float), np.arange(len(B), dtype=float)
    spl = RectBivariateSpline(ib, ia, Z, kx=2, ky=2)
    fa, fb = np.linspace(0, len(A) - 1, 50), np.linspace(0, len(B) - 1, 50)
    FA, FB = np.meshgrid(fa, fb); FZ = spl(fb, fa)
    s = ax.plot_surface(FA, FB, FZ, cmap=cm, edgecolor="none", alpha=.95, rstride=1, cstride=1)
    zmin = FZ.min() - .1 * (FZ.max() - FZ.min() + 1e-9)
    ax.contour(FA, FB, FZ, zdir="z", offset=zmin, cmap=cm, levels=10, linewidths=1); ax.set_zlim(zmin, FZ.max())
    ax.scatter(*np.meshgrid(ia, ib), Z, color="k", s=16, depthshade=False)
    for (va, vb), col, mk, sz in ((default, "red", "o", 90), (best, "limegreen", "*", 190)):
        if va in A and vb in B:
            ax.scatter([A.index(va)], [B.index(vb)], [Z[B.index(vb), A.index(va)]], color=col, marker=mk, s=sz, edgecolor="k", zorder=20, depthshade=False)
    ax.set_xticks(ia); ax.set_xticklabels([fmt[a].format(x) for x in A]); ax.set_yticks(ib); ax.set_yticklabels([fmt[b].format(x) for x in B])
    LB = {"kap": "Hệ số ngưỡng κ", "step": "Bước λ", "win": "Cửa sổ (gói)"}
    ax.set_xlabel(LB[a], labelpad=6); ax.set_ylabel(LB[b], labelpad=8); ax.zaxis.set_major_locator(plt.MaxNLocator(5)); ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.3g'))
    ax.set_title(title, pad=10); ax.view_init(27, -58); ax.set_box_aspect((1, 1, .8), zoom=1.15)
    return s


def draw(key, g):
    fmt = {"kap": "{:g}", "step": "{:g}", "win": "{:g}"}
    dflt, best_row = 1.5, g.loc[g.delay.idxmin()]
    fig = plt.figure(figsize=(17, 13.5))
    cfg = [(("win", "step"), "delay", "Blues", "Trễ trung bình (ms)", "Trễ theo cửa sổ và bước λ"),
           (("win", "step"), "loss", "Reds", "Mất gói (tắc nghẽn) (%)", "Mất gói theo cửa sổ và bước λ"),
           (("kap", "step"), "delay", "Oranges", "Trễ trung bình (ms)", "Trễ theo hệ số ngưỡng κ và bước λ"),
           (("kap", "step"), "loss", "Greens", "Mất gói (tắc nghẽn) (%)", "Mất gói theo hệ số ngưỡng κ và bước λ")]
    for i, ((a, b), z, cm, zl, ttl) in enumerate(cfg, 1):
        x0, y0 = (.02, .52) if i in (1, 3)[:0] or False else (0, 0)
        col, row = (i - 1) % 2, (i - 1) // 2
        x0, y0 = .01 + col * .49, .53 - row * .46
        ax = fig.add_axes([x0, y0, .40, .40], projection="3d")
        cax = fig.add_axes([x0 + .40, y0 + .06, .012, .28])
        mk = lambda v, c: (v[a] if False else None)
        d_ab = {"win": 500, "step": 0.05, "kap": dflt}; b_ab = {"win": best_row.win, "step": best_row.step, "kap": best_row.kap}
        s = surf(ax, g, a, b, z, cm, ttl, zl, (d_ab[a], d_ab[b]), (b_ab[a], b_ab[b]), fmt)
        cb = fig.colorbar(s, cax=cax)
        cb.set_label(zl, fontsize=11, labelpad=8); cb.ax.tick_params(labelsize=9)
    fig.suptitle(f"{NAME[key]}: ảnh hưởng của 3 biến của bộ tự điều chỉnh tự động λ (ctrl3) lên trễ và mất gói", fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .008, f"Mỗi bề mặt = trung bình theo biến còn lại; chấm đen = cấu hình đã chạy (n = {int(g.n.min())} seed); chấm đỏ = cấu hình đã chốt (bước 0,05; cửa sổ 500; κ = 1,5); sao xanh = cấu hình trễ thấp nhất. Phân tích thăm dò; ngưỡng D_ref tự suy ra (κ × trung vị trễ 5 cửa sổ đầu).",
             ha="center", fontsize=10, color="#666")
    return fig


# --- chay ngay khi nap file ---
figs = []
for key in ("ns3", "ukm"):
    g = load(key)
    if g is None: continue
    print(g.round(3).to_string(index=False))
    f = draw(key, g)
    if os.environ.get("SAVE_DIR"): f.savefig(os.path.join(os.environ["SAVE_DIR"], f"ctrl3_sweep3d_{key}.png"), dpi=110)
if not os.environ.get("SAVE_DIR"): plt.show()
