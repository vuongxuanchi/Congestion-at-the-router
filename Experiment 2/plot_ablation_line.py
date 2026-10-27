"""
Ve lai bieu do Thuc nghiem 2 (khu bien WCNN) dang DUONG (line) thay vi cot,
doc truc tiep tu ablation_summary.csv da co san (khong can train lai).
Nen xam nhat, co the sua duong dan CSV_PATH neu can.

Usage: python3 plot_ablation_line.py
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

CSV_PATH = "ablation_summary.csv"

VERSION_ORDER = ["v1_base", "v2_lr_sched", "v3_ema", "v4_multiwave"]
VERSION_LABEL = {
    "v1_base": "v1\nGốc",
    "v2_lr_sched": "v2\n+Tách LR/lịch",
    "v3_ema": "v3\n+EMA",
    "v4_multiwave": "v4\n+Wav-KAN đa TP",
}

df = pd.read_csv(CSV_PATH)
df["order"] = df["version"].map({v: i for i, v in enumerate(VERSION_ORDER)})
df = df.sort_values("order").reset_index(drop=True)

x = np.arange(len(df))
xticklabels = [VERSION_LABEL.get(v, v) for v in df["version"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

for ax in axes:
    ax.set_facecolor("#f0f0f0")           # nen xam nhat cho vung ve
    ax.grid(True, color="white", linewidth=1.3, zorder=0)  # luoi trang noi tren nen xam
    ax.set_axisbelow(True)
    ax.set_xlim(-0.4, len(x) - 0.6)        # them le trai/phai de nhan khong dung vien

# mau tuoi hon (xanh duong sang / do tuoi), khong doi cau truc ve
COLOR_BLUE = "#0072ff"
COLOR_RED = "#ff3b30"

# ---- Bieu do trai: Accuracy (duong, co errorbar std) ----
ax = axes[0]
ax.errorbar(x, df["accuracy_mean"], yerr=df["accuracy_std"], fmt="o-",
            color=COLOR_BLUE, linewidth=2.6, markersize=9, markerfacecolor="white",
            markeredgewidth=2, capsize=5, capthick=1.8, ecolor=COLOR_BLUE, zorder=3)
ax.fill_between(x, df["accuracy_mean"] - df["accuracy_std"], df["accuracy_mean"] + df["accuracy_std"],
                 color=COLOR_BLUE, alpha=0.10, zorder=2)
for xi, (m, s) in enumerate(zip(df["accuracy_mean"], df["accuracy_std"])):
    ax.annotate(f"{m:.4f}", (xi, m), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold", color=COLOR_BLUE)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=9.5)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("Accuracy qua từng bước khử biến", fontweight="bold", fontsize=13)
y0, y1 = df["accuracy_mean"].min() - 0.006, df["accuracy_mean"].max() + 0.006
ax.set_ylim(y0, y1)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

# ---- Bieu do phai: muc bat on (thang log de thay ro cu roi giam manh) ----
ax = axes[1]
ax.plot(x, df["val_loss_max_jump_mean"], "o-", color=COLOR_RED, linewidth=2.6,
         markersize=9, markerfacecolor="white", markeredgewidth=2, zorder=3)
ax.fill_between(x, 0, df["val_loss_max_jump_mean"], color=COLOR_RED, alpha=0.10, zorder=2)
for xi, v in enumerate(df["val_loss_max_jump_mean"]):
    ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold", color=COLOR_RED)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=9.5)
ax.set_ylabel("Dao động val_loss lớn nhất giữa 2 epoch", fontsize=10.5)
ax.set_title("Mức độ bất ổn huấn luyện", fontweight="bold", fontsize=13)
ax.set_ylim(-0.01, df["val_loss_max_jump_mean"].max() * 1.20)

plt.suptitle("Thực nghiệm 2 — Khử biến kiến trúc WCNN (trung bình ± độ lệch chuẩn qua 10 seed)",
             fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
