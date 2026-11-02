# -*- coding: utf-8 -*-
"""Phan tich HAM PHAN THUONG cua DN-dCoD (cong thuc 3.48) - CHI HIEN THI, KHONG LUU FILE.
  R = -( b1*min(D/D_max,1) + b2*Loss + b3*(1 - min(Thr/C,1)) ),  beta phu thuoc nhan goi (Mice/Sheep/Elephant)
  Hinh 1 (3D+2D): be mat R theo (tai rho, xac suat drop chu dong p) qua mo hinh hang doi M/M/1/K:
          Khong nhan(=Sheep) | Mice | Elephant | p toi uu theo tai cho 3 nhan (be mat CONG, co diem toi uu)
  Hinh 2 (2D 2x2): R theo do tre | R theo mat goi | R theo thong luong | "ty gia doi": do tre can giam / 1% mat goi them
Chay:  %run ve_phan_thuong_3d.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- tham so lay tu dcod_env.py (sua neu ban da doi) ---
BETA = {"Không nhãn / Sheep": (1.0, 1.0, 1.0), "Mice": (1.8, 0.6, 0.6), "Elephant": (0.6, 1.2, 1.2)}
DOMAINS = {"NS-3": dict(D_MAX=200.0, C=10.0), "UKM": dict(D_MAX=1000.0, C=2.0)}
THR_FIX = 0.95          # thong luong co dinh = 95% C khi ve be mat theo (do tre, mat goi)
CL = {"Không nhãn / Sheep": "#4C78A8", "Mice": "#E8743B", "Elephant": "#2E9E6B"}

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

def R(beta, D, loss, thr, D_MAX, C):
    b1, b2, b3 = beta
    return -(b1 * np.minimum(D / D_MAX, 1.0) + b2 * loss + b3 * (1.0 - np.minimum(thr / C, 1.0)))

# ---- mo hinh hang doi M/M/1/K de bien (p, tai) -> (tre, mat goi, thong luong) ----
PKT_BYTES, T_TARGET_MS = 1000, 50   # AQM giu hang doi quanh T_target (<= T_MAX = 50 ms trong dcod_env.py) -> dem hieu dung K = T*mu

def queue_metrics(rho, p, C):
    """rho = tai den / dung luong; p = xac suat DROP chu dong. Tra ve (tre ms, mat goi, thong luong Mbps)."""
    mu = C * 1e6 / (8 * PKT_BYTES)                       # goi/s
    re = np.clip(rho * (1 - p), 1e-6, None)              # tai hieu dung sau khi drop chu dong
    re = np.where(np.abs(re - 1) < 1e-4, 1 + 1e-4, re)
    K = max(4, int(round(T_TARGET_MS / 1000.0 * mu)))
    with np.errstate(over="ignore", invalid="ignore"):
        lr = np.log(re)
        pb = (1 - re) * np.exp(K * lr) / (1 - np.exp((K + 1) * lr))
        L = re / (1 - re) - (K + 1) * np.exp((K + 1) * lr) / (1 - np.exp((K + 1) * lr))
    pb = np.nan_to_num(pb, nan=1.0, posinf=1.0, neginf=1.0).clip(0, 1)
    acc = re * (1 - pb)                                   # tai duoc nhan / mu
    delay_ms = np.nan_to_num(L / np.maximum(acc, 1e-9) / mu * 1000, nan=0.0, posinf=1e9)
    loss = p + (1 - p) * pb                               # tong mat goi (chu dong + tran dem)
    thr = np.minimum(rho * (1 - loss), 1.0) * C
    return delay_ms, loss, thr

def fig_3d(dom, p):
    D_MAX, C = p["D_MAX"], p["C"]
    rho = np.linspace(.8, 2.0, 80); pd_ = np.linspace(0, .60, 80); X, P = np.meshgrid(rho, pd_)
    Dm, Ls, Th = queue_metrics(X, P, C)
    Z = {k: R(b, Dm, Ls, Th, D_MAX, C) for k, b in BETA.items()}
    Z["diff"] = Z["Mice"] - Z["Elephant"]
    mu = C * 1e6 / (8 * PKT_BYTES); Keff = max(4, int(round(T_TARGET_MS / 1000.0 * mu)))
    fig = plt.figure(figsize=(16, 13.5))
    cfg = [("Không nhãn / Sheep", "viridis"), ("Mice", "magma"), ("Elephant", "cividis"), ("diff", "coolwarm")]
    for i, (k, cmap) in enumerate(cfg, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d"); z = Z[k]
        kw = {}
        if k == "diff": m = np.abs(z).max(); kw = dict(vmin=-m, vmax=m)
        s = ax.plot_surface(X, P * 100, z, cmap=cmap, edgecolor="none", alpha=.97, antialiased=True, rstride=1, cstride=1, **kw)
        zmin = z.min() - .10 * (z.max() - z.min() + 1e-9)
        ax.contour(X, P * 100, z, zdir="z", offset=zmin, cmap=cmap, levels=12, linewidths=1, **kw)
        ax.set_zlim(zmin, z.max())
        if k != "diff":                                            # duong p* (toi uu) theo tai, ve tren be mat
            ib = np.argmax(z, axis=0)
            ax.plot(rho, P[ib, np.arange(len(rho))] * 100, z[ib, np.arange(len(rho))], color="red", lw=3, zorder=10, label="p tối ưu")
        ax.set_xlabel("Tải / dung lượng (ρ)", labelpad=10); ax.set_ylabel("Drop chủ động p (%)", labelpad=10)
        ax.set_xticks([1.0, 1.5, 2.0]); ax.set_yticks([0, 20, 40, 60])
        if k == "diff": ax.set_title("Chênh lệch R(Mice) − R(Elephant)\n(đỏ: Mice lợi hơn; xanh: Elephant lợi hơn)", pad=14)
        else:
            b = BETA[k]; ax.set_title(f"{k}\nβ=({b[0]}, {b[1]}, {b[2]})   (đường đỏ = p tối ưu theo tải)", pad=14)
        ax.view_init(elev=27, azim=-58); ax.set_box_aspect((1, 1, .8), zoom=1.08)
        cb = fig.colorbar(s, ax=ax, shrink=.6, pad=.07, aspect=18, fraction=.03)
        cb.set_label("Reward" if k != "diff" else "ΔReward", labelpad=8)
    fig.suptitle(f"{dom}: phần thưởng theo tải và mức drop chủ động (hàng đợi M/M/1/K, K≈{Keff} gói ứng với T_target={T_TARGET_MS} ms, D_max={D_MAX:g} ms)",
                 fontsize=15, fontweight="bold", y=.985)
    fig.text(.5, .015, "Mô hình minh họa: lưu lượng Poisson, gói 1000 byte; đường đỏ = mức drop chủ động p cho reward cao nhất ở mỗi mức tải", ha="center", va="bottom", fontsize=10, color="#666")
    fig.subplots_adjust(left=-.03, right=.97, top=.92, bottom=.12, wspace=-.10, hspace=.26); plt.show()

def fig_2d(dom, p):
    D_MAX, C = p["D_MAX"], p["C"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6)); a1, a2, a3, a4 = ax.ravel()
    d = np.linspace(0, D_MAX, 200); l = np.linspace(0, .20, 200); t = np.linspace(0, C, 200)
    for k, b in BETA.items():
        a1.plot(d, R(b, d, .02, .95 * C, D_MAX, C), color=CL[k], lw=2.6, label=k)
        a2.plot(l * 100, R(b, D_MAX * .1, l, .95 * C, D_MAX, C), color=CL[k], lw=2.6, label=k)
        a3.plot(t, R(b, D_MAX * .1, .02, t, D_MAX, C), color=CL[k], lw=2.6, label=k)
    a1.set_title("Reward theo độ trễ\n(mất gói 2%, thông lượng 95% C)"); a1.set_xlabel("Độ trễ (ms)"); a1.set_ylabel("Reward")
    a2.set_title(f"Reward theo mất gói\n(độ trễ {D_MAX*.1:g} ms, thông lượng 95% C)"); a2.set_xlabel("Mất gói (%)")
    a3.set_title(f"Reward theo thông lượng\n(độ trễ {D_MAX*.1:g} ms, mất gói 2%)"); a3.set_xlabel("Thông lượng (Mbps)")
    for a in (a1, a2, a3): a.grid(alpha=.25); a.legend(fontsize=10)
    # ty gia doi: bo them 1% goi thi can giam bao nhieu ms do tre trung binh de hoa von
    ks = list(BETA); be = [(b[1] / b[0]) * D_MAX * .01 for b in BETA.values()]
    bars = a4.bar(range(3), be, color=[CL[k] for k in ks], width=.55)
    for r, v in zip(bars, be): a4.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f} ms", ha="center", va="bottom", fontweight="bold")
    a4.set_xticks(range(3)); a4.set_xticklabels(ks); a4.set_ylabel("ms")
    a4.set_title("Tỷ giá đổi: độ trễ cần giảm để bù 1% mất gói thêm\n(β₂/β₁ · D_max · 1%) — thấp = dễ chấp nhận drop")
    a4.grid(alpha=.25, axis="y")
    fig.suptitle(f"{dom}: độ nhạy của hàm phần thưởng theo từng nhãn", fontsize=15, fontweight="bold", y=.93)
    fig.tight_layout(rect=(0, 0, 1, .90)); plt.show()

for dom, p in DOMAINS.items():
    fig_3d(dom, p)
    fig_2d(dom, p)
