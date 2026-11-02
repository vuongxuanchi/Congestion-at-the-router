"""
Ve lai bieu do hoi tu reward + do tre/mat goi qua episode CHI TU FILE LOG
DA CO SAN (results/dn_dcod/{job_name}_log.json) - KHONG train lai.

Thay doi so voi ban trong train_dqn_dcod.py:
  - Duong "Mat goi (%)": net LIEN (khong con net dut), do day 2.2
  - Tieu de dung dung ten tieng Viet: DN-dCoD (Khong nhan) / DN-dCoD (Nhung nhan)

Usage:
  python3 plot_dn_dcod_from_log.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
RESULTS_DIR = f"{DATA_DIR}/results/dn_dcod"

DISPLAY_NAME = {
    "dn_dcod_khong_nhan": "DN-dCoD (Không nhãn)",
    "dn_dcod": "DN-dCoD (Nhúng nhãn)",
}


def plot_from_log(job_name, display_name):
    log_path = Path(RESULTS_DIR) / f"{job_name}_log.json"
    if not log_path.exists():
        print(f"[warn] khong tim thay {log_path} -> bo qua {display_name}")
        return

    result = json.loads(log_path.read_text())
    episode_rewards = result["episode_rewards"]
    episode_summaries = result["episode_summaries"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.set_facecolor("#f0f0f0")
        ax.grid(True, color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)

    ep = np.arange(1, len(episode_rewards) + 1)
    axes[0].plot(ep, episode_rewards, color="#2980b9", linewidth=2.5, marker="o", markersize=4)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Tổng reward")
    axes[0].set_title(f"Hội tụ reward — {display_name}", fontweight="bold")

    delays = [s["avg_delay_ms"] for s in episode_summaries]
    losses_pct = [s["loss_rate"] * 100 for s in episode_summaries]
    ax2 = axes[1]
    ax2.plot(ep, delays, color="#c0392b", linewidth=2.5, label="Độ trễ (ms)")
    ax2b = ax2.twinx()
    ax2b.plot(ep, losses_pct, color="#8e44ad", linewidth=2.2, linestyle="-", label="Mất gói (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Độ trễ (ms)", color="#c0392b")
    ax2b.set_ylabel("Mất gói (%)", color="#8e44ad")
    ax2.set_title(f"Độ trễ & mất gói qua các episode — {display_name}", fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for job_name, display_name in DISPLAY_NAME.items():
        plot_from_log(job_name, display_name)
