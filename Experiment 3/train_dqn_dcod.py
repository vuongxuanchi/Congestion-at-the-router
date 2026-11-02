"""
Huan luyen DQN cho DN-dCoD (khong nhan, Chuong 3 - phien ban khu bien) hoac
DN-dCoD (co nhan FRCNC, dat --label_mode external + set_label_provider rieng
o mot script noi tiep sau). Dung DCoDEnv (dcod_env.py) lam moi truong, replay
tren vet DropTail.

Kien truc DQN: MLP 2 lop an, replay buffer, epsilon-greedy giam dan, target
network cap nhat dinh ky (giong DQN co dien - Mnih 2015), toi uu bang Adam +
Huber loss (on dinh hon MSE voi outlier reward).

Usage:
  python3 train_dqn_dcod.py --n_episodes 30 --steps_per_episode 5000
"""
import os
import sys
import json
import random
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# DATA_DIR phai duoc xac dinh TRUOC khi import dcod_env, vi day cung la
# noi tim file dcod_env.py (Python KHONG tu tim module trong thu muc Drive -
# chi tim trong thu muc lam viec hien tai hoac cac duong dan da khai bao
# trong sys.path). Them DATA_DIR vao sys.path o day de "from dcod_env
# import ..." chay duoc DU dcod_env.py dang nam o Drive (data1) hay o
# /content/ - khong con bi "ModuleNotFoundError: No module named 'dcod_env'"
# du dcod_env.py da duoc upload dung cho o DATA_DIR.
DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()  # ep Python quet lai thu muc DATA_DIR - tranh loi
                                 # ModuleNotFoundError khi file .py duoc upload len Drive
                                 # SAU KHI phien Colab da tung quet thu muc nay 1 lan

from dcod_env import DCoDEnv, ACTIONS

RESULTS_DIR = f"{DATA_DIR}/results/dn_dcod"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAMMA = 0.95
LR = 1e-3
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 20000
TARGET_UPDATE_EVERY = 500        # so buoc (step), khong phai episode
MIN_BUFFER_BEFORE_TRAIN = 500

print(f"Device: {DEVICE}")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.action_dim = action_dim
        self.global_step = 0

    def epsilon(self):
        frac = min(1.0, self.global_step / EPS_DECAY_STEPS)
        return EPS_START + frac * (EPS_END - EPS_START)

    def select_action(self, state, greedy=False):
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < max(BATCH_SIZE, MIN_BUFFER_BEFORE_TRAIN):
            return None
        s, a, r, s2, d = self.buffer.sample(BATCH_SIZE)
        s = torch.tensor(s, device=DEVICE)
        a = torch.tensor(a, device=DEVICE)
        r = torch.tensor(r, device=DEVICE)
        s2 = torch.tensor(s2, device=DEVICE)
        d = torch.tensor(d, device=DEVICE)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(s2).max(dim=1)[0]
            target = r + GAMMA * next_q * (1.0 - d)
        loss = F.smooth_l1_loss(q_values, target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

        self.global_step += 1
        if self.global_step % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


def train(droptail_csv, n_episodes, steps_per_episode, direction, label_mode, job_name):
    env = DCoDEnv(droptail_csv, direction=direction, label_mode=label_mode,
                  max_steps=None)
    agent = DQNAgent(env.state_dim, env.action_dim)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(RESULTS_DIR) / f"{job_name}_checkpoint.pt"

    episode_rewards, episode_losses_avg, episode_summaries = [], [], []
    start_episode = 0

    # --- RESUME: neu da co checkpoint tu lan chay truoc, nap lai thay vi
    # train tu dau - giong co che resume da dung cho CNN/WCNN/FRCNC. Luu y:
    # KHONG luu lai replay buffer (se rat lon) - buffer bat dau lai tu rong
    # sau resume, nhung Q-network/target-network/optimizer/global_step
    # (dieu khien epsilon) va lich su reward/loss deu duoc giu nguyen, nen
    # qua trinh hoc tiep tuc dung mach, khong mat tien do da hoc duoc.
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE)
        try:
            agent.q_net.load_state_dict(ck["q_net_state"])
            agent.target_net.load_state_dict(ck["target_net_state"])
            agent.opt.load_state_dict(ck["opt_state"])
            agent.global_step = ck["global_step"]
            episode_rewards = ck["episode_rewards"]
            episode_losses_avg = ck["episode_losses_avg"]
            episode_summaries = ck["episode_summaries"]
            start_episode = ck["episode"] + 1
            print(f"[{job_name}] resuming from episode {start_episode} "
                  f"(da hoan thanh {start_episode}/{n_episodes} episode truoc do)")
        except (RuntimeError, KeyError) as e:
            print(f"[{job_name}] WARNING: checkpoint khong khop (kien truc/state_dim thay doi?) "
                  f"-> train lai tu dau.\n   ({e})")
            start_episode = 0
            episode_rewards, episode_losses_avg, episode_summaries = [], [], []

    if start_episode >= n_episodes:
        print(f"[{job_name}] da hoan thanh du {n_episodes} episode tu truoc -> bo qua train, "
              f"chi chay lai buoc danh gia greedy.")

    for ep in range(start_episode, n_episodes):
        s = env.reset()
        ep_reward, ep_losses, n_steps = 0.0, [], 0
        max_steps = min(steps_per_episode, env.n)
        for t in range(max_steps):
            a = agent.select_action(s)
            s2, r, done, info = env.step(a)
            agent.store(s, a, r, s2, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            s = s2
            ep_reward += r
            n_steps += 1
            if done:
                break

        summary = env.episode_summary()
        episode_rewards.append(ep_reward)
        episode_losses_avg.append(float(np.mean(ep_losses)) if ep_losses else float("nan"))
        episode_summaries.append(summary)
        print(f"[{job_name}] episode {ep+1}/{n_episodes} steps={n_steps} "
              f"reward={ep_reward:.2f} eps={agent.epsilon():.3f} "
              f"loss={episode_losses_avg[-1]:.4f} "
              f"delay={summary['avg_delay_ms']:.2f}ms loss_rate={summary['loss_rate']*100:.2f}% "
              f"throughput={summary['throughput_mbps']:.2f}Mbps T_target={summary['final_T_target_ms']:.2f}ms")

        # luu checkpoint SAU MOI episode - neu Colab bi ngat giua chung, chay
        # lai dung lenh cu se tiep tuc tu episode ke tiep, khong mat cong da hoc.
        torch.save({
            "q_net_state": agent.q_net.state_dict(),
            "target_net_state": agent.target_net.state_dict(),
            "opt_state": agent.opt.state_dict(),
            "global_step": agent.global_step,
            "episode": ep,
            "episode_rewards": episode_rewards,
            "episode_losses_avg": episode_losses_avg,
            "episode_summaries": episode_summaries,
        }, ckpt_path)

    result = dict(job_name=job_name, label_mode=label_mode, n_episodes=n_episodes,
                  episode_rewards=episode_rewards, episode_losses=episode_losses_avg,
                  episode_summaries=episode_summaries,
                  final_summary=episode_summaries[-1] if episode_summaries else None)
    log_path = Path(RESULTS_DIR) / f"{job_name}_log.json"
    log_path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {log_path}")

    ckpt_path = Path(RESULTS_DIR) / f"{job_name}_qnet.pt"
    torch.save({"model_state": agent.q_net.state_dict(), "state_dim": env.state_dim,
                "action_dim": env.action_dim, "label_mode": label_mode}, ckpt_path)
    print(f"saved -> {ckpt_path}")

    return result, agent, env


def plot_training_curve(result, job_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in axes:
        ax.set_facecolor("#f0f0f0")
        ax.grid(True, color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)

    ep = np.arange(1, len(result["episode_rewards"]) + 1)
    axes[0].plot(ep, result["episode_rewards"], color="#2980b9", linewidth=2.5, marker="o", markersize=4)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Tổng reward")
    axes[0].set_title(f"Hội tụ reward — {job_name}", fontweight="bold")

    delays = [s["avg_delay_ms"] for s in result["episode_summaries"]]
    losses_pct = [s["loss_rate"] * 100 for s in result["episode_summaries"]]
    ax2 = axes[1]
    ax2.plot(ep, delays, color="#c0392b", linewidth=2.5, label="Độ trễ (ms)")
    ax2b = ax2.twinx()
    ax2b.plot(ep, losses_pct, color="#8e44ad", linewidth=2.5, linestyle="--", label="Mất gói (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Độ trễ (ms)", color="#c0392b")
    ax2b.set_ylabel("Mất gói (%)", color="#8e44ad")
    ax2.set_title(f"Độ trễ & mất gói qua các episode — {job_name}", fontweight="bold")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--steps_per_episode", type=int, default=5000)
    ap.add_argument("--direction", type=int, default=0)
    ap.add_argument("--label_mode", type=str, default="none", choices=["none", "external"],
                     help="'none' = DN-dCoD (khong nhan, buoc hien tai); 'external' = DN-dCoD "
                          "(can goi env.set_label_provider() truoc, o script noi tiep)")
    ap.add_argument("--job_name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run", type=int, default=1)
    cli_args, _unknown = ap.parse_known_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    droptail_csv = f"{DATA_DIR}/out_ip_header_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"
    # Dung nhat ten theo luan van (Bang 4.8): "DQN-dCoD" va "DN-dCoD" la
    # CUNG MOT ho ten "DN-dCoD", chi khac o "(khong nhan)" - khong phai 2
    # ten viet tat khac nhau cho 2 mo hinh khac nhau ve ban chat.
    job_name = cli_args.job_name or ("dn_dcod_khong_nhan" if cli_args.label_mode == "none" else "dn_dcod")

    result = train(droptail_csv, cli_args.n_episodes, cli_args.steps_per_episode,
                    cli_args.direction, cli_args.label_mode, job_name)
    result, agent, env = result
    plot_training_curve(result, job_name)

    # chay 1 lan GREEDY (khong explore) tren toan bo du lieu, xuat dung
    # schema out_ip_header_*.csv de dung TRUC TIEP voi analyze_aqm_comparison.py
    # (Thuc nghiem 3) - vd them "dn_dcod_khong_nhan" vao AQM_AVAILABLE cua script do.
    export_csv = f"{DATA_DIR}/out_ip_header_{job_name}_seed{cli_args.seed}_run{cli_args.run}.csv"
    final_summary = env.run_greedy_and_export(agent, export_csv)
    print(f"\n[greedy eval] {final_summary}")
