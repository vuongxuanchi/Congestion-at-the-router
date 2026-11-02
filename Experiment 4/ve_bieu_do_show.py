# -*- coding: utf-8 -*-
"""Ve bieu do (CHI HIEN THI, KHONG LUU FILE):
  (A) 2x2 line: Tre TB, Mat goi %, Thong luong, Jain  -- 10 seed, cho NS-3 va UKM
  (B) 2x2 theo 300 episode: Tre, Mat goi, Thong luong, Reward (DQN train)  -- NS-3 va UKM
  (C) 1x2: Hoi tu reward train (DQN)  |  Reward validation greedy (DQN)  -- NS-3 va UKM, mau rieng
Chay trong Colab:  %run ve_bieu_do_show.py   (hoac dan noi dung vao 1 o)
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

D = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
SEARCH = [D, D + "/data2"]                       # noi co aqm_wide_*.csv

# ---- ten thu muc ket qua huan luyen (sua neu khac) ----
REWARD_DIR = {"NS-3": f"{D}/results/dn_dcod_v2", "UKM": f"{D}/results_ukm/dn_dcod_v4"}
SUFFIX = {"NS-3": "_offline", "UKM": "_ukm"}

ORDER = ["dcodel", "dn_dcod_khong_nhan", "dn_dcod"]
LBL = {"dcodel": "dCoDel", "dn_dcod_khong_nhan": "DN-dCoD (không nhãn)", "dn_dcod": "DN-dCoD (nhúng nhãn)"}
COL = {"dcodel": "blue", "dn_dcod_khong_nhan": "orange", "dn_dcod": "red"}
MK = {"dcodel": "o", "dn_dcod_khong_nhan": "s", "dn_dcod": "D"}
METRICS = [("avg_delay_ms", "Độ trễ TB (ms)", 1), ("loss_rate", "Tỷ lệ mất gói (%)", 100),
           ("throughput_mbps", "Thông lượng (Mbps)", 1), ("jain_index", "Chỉ số Jain", 1)]
LW = 2.2          # do day line cua cac hinh 2x2 (mong hon)
LW2 = 3.4         # do day line cua hinh 1x2
# mau rieng cho hinh 1x2 (khac mau 2x2)
COL2 = {"dn_dcod_khong_nhan": "#8E44AD", "dn_dcod": "#16A085"}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True, "legend.frameon": False})

def find(name):
    for b in SEARCH:
        p = os.path.join(b, name)
        if os.path.exists(p):
            return p
    return None

# ======================= (A) 2x2 : 4 chi so qua 10 seed =======================
def plot_metrics(dom):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    got = False
    for ax, (m, title, f) in zip(axes.ravel(), METRICS):
        p = find(f"aqm_wide_{m}{SUFFIX[dom]}.csv")
        if p is None:
            ax.set_title(title + " (thiếu file)"); continue
        df = pd.read_csv(p, index_col=0)[ORDER].dropna() * f
        x = np.arange(1, len(df) + 1)
        for k in ORDER:
            ax.plot(x, df[k].values, color=COL[k], lw=LW, marker=MK[k], ms=7, mec="white", mew=1.2, label=LBL[k])
        ax.set_title(title); ax.set_xlabel("Lần chạy (seed)"); ax.set_xticks(x)
        got = True
    if not got:
        print(f"[{dom}] khong thay aqm_wide_*{SUFFIX[dom]}.csv trong {SEARCH}"); plt.close(fig); return
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(f"{dom}: so sánh 3 cơ chế offline qua 10 seed", y=1.045, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); plt.show()

# ================== (C) 1x2 : hoi tu reward | do chinh xac TB ==================
def smooth(y, w=10):
    return pd.Series(y).rolling(w, min_periods=1).mean().values

# ============ (B) 2x2 theo 300 episode (train) ============
def plot_episodes(dom):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5)); got = False
    spec = [("avg_delay_ms", "Độ trễ TB (ms)", 1), ("loss_rate", "Tỷ lệ mất gói (%)", 100),
            ("throughput_mbps", "Thông lượng (Mbps)", 1), ("reward", "Reward (train)", 1)]
    for ax, (key, title, f) in zip(axes.ravel(), spec):
        for job in ["dn_dcod_khong_nhan", "dn_dcod"]:
            p = f"{REWARD_DIR[dom]}/{job}_v2_log.json"
            if not os.path.exists(p): continue
            lg = json.load(open(p))
            y = (np.array(lg["episode_rewards"], float) if key == "reward"
                 else np.array([sm[key] for sm in lg["episode_summaries"]], float) * f)
            ep = np.arange(1, len(y) + 1)
            ax.plot(ep, y, color=COL[job], lw=.8, alpha=.2)
            ax.plot(ep, smooth(y), color=COL[job], lw=LW, label=LBL[job] + " (TB trượt 10)"); got = True
        ax.set_title(title); ax.set_xlabel("Episode")
    if not got:
        print(f"[{dom}] thieu log trong {REWARD_DIR[dom]}"); plt.close(fig); return
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=12, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(f"{dom}: huấn luyện DN-dCoD qua 300 episode", y=1.045, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); plt.show()

def plot_conv(dom):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    # --- trai: hoi tu reward ---
    for job, k in [("dn_dcod_khong_nhan", "dn_dcod_khong_nhan"), ("dn_dcod", "dn_dcod")]:
        p = f"{REWARD_DIR[dom]}/{job}_v2_log.json"
        if not os.path.exists(p):
            print(f"[{dom}] thieu {p}"); continue
        lg = json.load(open(p)); r = np.array(lg["episode_rewards"], float); ep = np.arange(1, len(r) + 1)
        a1.plot(ep, r, color=COL2[k], lw=1, alpha=.22)
        a1.plot(ep, smooth(r), color=COL2[k], lw=LW2, label=LBL[k] + " (TB trượt 10)")
        b = lg.get("best_episode")
        if b: a1.axvline(b, color=COL2[k], ls=":", lw=1.8, alpha=.9)
    a1.set_title(f"Hội tụ reward — {dom}"); a1.set_xlabel("Episode"); a1.set_ylabel("Tổng reward")
    a1.legend(fontsize=10)
    # --- phai: reward validation (greedy, tren cac doan co dinh) ---
    for job in ["dn_dcod_khong_nhan", "dn_dcod"]:
        p = f"{REWARD_DIR[dom]}/{job}_v2_log.json"
        if not os.path.exists(p): continue
        vh = json.load(open(p)).get("val_history", [])
        if not vh: continue
        e = [v["episode"] for v in vh]; r = np.array([v["reward"] for v in vh], float)
        a2.plot(e, r, color=COL2[job], lw=LW2, marker="o", ms=6, mec="white", mew=1, label=LBL[job])
        b = int(np.argmax(r)); a2.scatter(e[b], r[b], s=260, marker="*", color=COL2[job], edgecolor="black", zorder=5)
    a2.set_title(f"Reward validation (greedy) — {dom}"); a2.set_xlabel("Episode"); a2.set_ylabel("Reward trung bình")
    if a2.get_legend_handles_labels()[0]: a2.legend(fontsize=10, title="★ = checkpoint được chọn", title_fontsize=9)
    fig.tight_layout(); plt.show()

for dom in ["NS-3", "UKM"]:
    plot_metrics(dom)
for dom in ["NS-3", "UKM"]:
    plot_episodes(dom)
for dom in ["NS-3", "UKM"]:
    plot_conv(dom)