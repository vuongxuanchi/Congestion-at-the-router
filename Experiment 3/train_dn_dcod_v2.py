# -*- coding: utf-8 -*-
"""
Huan luyen LAI DN-dCoD (khong nhan / nhung nhan) de THEO DOI HOI TU, dung chung cho NS-3 va UKM.

Khac ban cu (train_dqn_dcod.py / run_ukm_dcod.py::stage_train):
  1. Moi episode la mot DOAN NGAU NHIEN cua vet (ban cu cho NS-3 luon bat dau tu goi dau tien,
     tuc chi hoc tren ~5000 goi dau cua vet).
  2. Epsilon giam CHAM (mac dinh 150.000 buoc, ban cu 20.000 buoc ~ 4 episode).
  3. Cu moi --val_every episode chay GREEDY tren --val_k doan CO DINH (rai deu theo vet), ghi
     reward/mat goi/tre; luu chinh sach TOT NHAT theo reward validation (khong dung chinh sach cuoi).
  4. Ghi log + ve duong cong hoi tu (PNG) de kiem tra huan luyen co hoi tu hay khong.
  5. --train_seed: huan luyen nhieu seed de do phuong sai do huan luyen (ten job them hau to _ts{k}).

Dau ra trong --out_dir:
  {job}_qnet.pt        chinh sach TOT NHAT (cung dinh dang cu: model_state, state_dim, action_dim, label_mode)
  {job}_last_qnet.pt   chinh sach CUOI
  {job}_v2_log.json    reward tung episode + lich su validation + cau hinh
  {job}_convergence.png
Co the resume: chay lai dung lenh cu se tiep tuc tu episode da luu.

Usage (Colab):
  !python3 train_dn_dcod_v2.py --domain ns3 --label_mode none      --n_episodes 300
  !python3 train_dn_dcod_v2.py --domain ns3 --label_mode external  --n_episodes 300
  !python3 train_dn_dcod_v2.py --domain ukm --label_mode none      --n_episodes 300
  !python3 train_dn_dcod_v2.py --domain ukm --label_mode external  --n_episodes 300
Tuy chon: --d_max_ms 200 (UKM, phat tre manh hon), --train_seed 1 2 ... (chay lan luot tung seed).
"""
import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
TRACE_DIR = os.environ.get("LUANAN_TRACE_DIR", DATA_DIR)
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()

import dcod_env
from dcod_env import DCoDEnv

PRESETS = {
    "ns3": dict(trace=f"{TRACE_DIR}/out_ip_header_droptail_seed10_run1.csv",
                labels=f"{TRACE_DIR}/frcnc_labels_droptail_seed10_run1.csv",
                rate_mbps=10.0, d_max_ms=200.0, cong_frac=0.0,
                out_dir=f"{DATA_DIR}/results/dn_dcod_v2", steps=5000),
    "ukm": dict(trace=f"{DATA_DIR}/out_ip_header_ukm_droptail_seed91_run0.csv",
                labels=f"{DATA_DIR}/frcnc_labels_ukm_seed91_run0.csv",
                rate_mbps=2.0, d_max_ms=1000.0, cong_frac=0.7,
                out_dir=f"{DATA_DIR}/results_ukm/dn_dcod_v4", steps=6000),
}


class SegEnv(DCoDEnv):
    """DCoDEnv + episode la mot doan ngau nhien (hoac doan co dinh khi validation)."""
    seg_len = None
    cong_starts = None
    cong_frac = 0.0
    fixed_start = None

    def set_congested_starts(self, rate_mbps):
        ts = self.arrivals["Timestamp"].to_numpy()
        sec = np.floor(ts).astype(int)
        load = np.bincount(sec, weights=self.arrivals["payload_size"].to_numpy(float)) * 8 / 1e6
        cong_sec = load > 0.7 * rate_mbps
        self.cong_starts = np.where(cong_sec[sec])[0]
        print(f"[env] {int(cong_sec.sum())}/{len(load)} giay nghen -> {len(self.cong_starts)} diem bat dau episode")

    def reset(self):
        s = super().reset()
        if self.seg_len and self.n > self.seg_len:
            hi = self.n - self.seg_len
            if self.fixed_start is not None:
                st = int(min(self.fixed_start, hi))
            elif self.cong_starts is not None and len(self.cong_starts) and random.random() < self.cong_frac:
                st = int(min(random.choice(self.cong_starts), hi))
            else:
                st = random.randint(0, hi)
            self.i = st
            ts = self.arrivals["Timestamp"]
            self.episode_start_s = float(ts.iloc[st])
            self.episode_end_s = float(ts.iloc[st + self.seg_len - 1])
            s = self._current_state()
        return s


def build_env(trace, label_mode, labels, rate_mbps, d_max_ms, seg_len, cong_frac):
    dcod_env.BOTTLENECK_RATE_MBPS = rate_mbps
    dcod_env.D_MAX_MS = d_max_ms
    env = SegEnv(trace, direction=0, label_mode=label_mode, max_steps=None)
    if label_mode == "external":
        import run_dn_dcod_labeled as rl
        cands = [labels, f"{DATA_DIR}/{Path(labels).name}", f"{TRACE_DIR}/{Path(labels).name}"]
        found = next((c for c in cands if Path(c).exists()), None)
        if found is None:
            raise FileNotFoundError("Chua co file nhan, da tim: " + " | ".join(cands) +
                                    "\n  -> truyen --labels <duong dan> hoac tao bang compute_frcnc_labels.py")
        print(f"[label] dung bang nhan: {found}")
        env.set_label_provider(rl.build_label_provider(found))
    env.seg_len = seg_len
    env.cong_frac = cong_frac
    if cong_frac > 0:
        env.set_congested_starts(rate_mbps)
    return env


def validate(env, agent, starts, seg_len):
    """Chay GREEDY tren cac doan co dinh; tra ve reward/mat goi/tre/thong luong trung binh."""
    R, L, D, T = [], [], [], []
    for st in starts:
        env.fixed_start = int(st)
        s = env.reset()
        tot = 0.0
        for _ in range(seg_len):
            a = agent.select_action(s, greedy=True)
            s, r, done, _ = env.step(a)
            tot += r
            if done:
                break
        sm = env.episode_summary()
        R.append(tot); L.append(sm["loss_rate"]); D.append(sm["avg_delay_ms"]); T.append(sm["throughput_mbps"])
    env.fixed_start = None
    return dict(reward=float(np.mean(R)), loss_rate=float(np.mean(L)),
                avg_delay_ms=float(np.mean(D)), throughput_mbps=float(np.mean(T)))


def plot_convergence(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(log["episode_rewards"]) + 1)
    r = np.array(log["episode_rewards"], dtype=float)
    w = 10
    roll = np.convolve(r, np.ones(w) / w, mode="valid") if len(r) >= w else r
    vh = log["val_history"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ep, r, color="#bbbbbb", lw=1, label="reward mỗi episode (train)")
    if len(r) >= w:
        ax[0].plot(ep[w - 1:], roll, color="#4C78A8", lw=2.2, label=f"trung bình trượt {w}")
    if vh:
        ax[0].plot([v["episode"] for v in vh], [v["reward"] for v in vh], "-o", color="#F58518", lw=2, ms=5,
                   label="reward validation (greedy)")
        b = log["best_episode"]
        bv = [v["reward"] for v in vh if v["episode"] == b]
        if bv:
            ax[0].scatter([b], bv, marker="*", s=220, color="#54A24B", zorder=5, label=f"tốt nhất (ep {b})")
    ax[0].set_title("Hội tụ: reward", fontweight="bold"); ax[0].set_xlabel("Episode"); ax[0].legend(fontsize=8)
    if vh:
        x = [v["episode"] for v in vh]
        ax[1].plot(x, [v["loss_rate"] * 100 for v in vh], "-o", color="#E45756")
        ax[1].set_title("Validation: tỷ lệ mất gói (%)", fontweight="bold"); ax[1].set_xlabel("Episode")
        ax[2].plot(x, [v["avg_delay_ms"] for v in vh], "-o", color="#72B7B2")
        ax[2].set_title("Validation: trễ TB (ms, theo env)", fontweight="bold"); ax[2].set_xlabel("Episode")
    for a in ax: a.grid(alpha=.25)
    fig.suptitle(f"{log['job']} — {log['config']['domain']} (seed huấn luyện {log['config']['train_seed']})", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def train(a, train_seed):
    import torch
    import train_dqn_dcod as tdd
    p = PRESETS[a.domain]
    trace = a.trace or p["trace"]
    labels = a.labels or p["labels"]
    rate = a.rate_mbps if a.rate_mbps is not None else p["rate_mbps"]
    d_max = a.d_max_ms if a.d_max_ms is not None else p["d_max_ms"]
    steps = a.steps or p["steps"]
    out_dir = Path(a.out_dir or p["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    base = "dn_dcod_khong_nhan" if a.label_mode == "none" else "dn_dcod"
    job = base + (f"_ts{train_seed}" if train_seed else "")

    random.seed(train_seed); np.random.seed(train_seed); torch.manual_seed(train_seed)
    tdd.EPS_DECAY_STEPS = a.eps_decay_steps
    tdd.EPS_END = a.eps_end

    env = build_env(trace, a.label_mode, labels, rate, d_max, steps, p["cong_frac"] if a.cong_frac is None else a.cong_frac)
    hi = max(0, env.n - steps)
    val_starts = [int(x) for x in np.linspace(0, hi, a.val_k)]
    agent = tdd.DQNAgent(env.state_dim, env.action_dim)
    cfg = dict(domain=a.domain, trace=str(trace), labels=str(labels) if a.label_mode == "external" else None,
               label_mode=a.label_mode, rate_mbps=rate, d_max_ms=d_max, steps_per_episode=steps,
               n_episodes=a.n_episodes, eps_decay_steps=a.eps_decay_steps, eps_end=a.eps_end,
               val_every=a.val_every, val_k=a.val_k, val_starts=val_starts, train_seed=train_seed,
               cong_frac=env.cong_frac, n_packets=int(env.n))
    print(f"[{job}] cau hinh: {cfg}")

    ck_path = out_dir / f"{job}_v2_checkpoint.pt"
    log = dict(job=job, config=cfg, episode_rewards=[], episode_summaries=[], val_history=[],
               best_val_reward=-1e18, best_episode=0)
    start = 0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=tdd.DEVICE)
        try:
            agent.q_net.load_state_dict(ck["q_net_state"]); agent.target_net.load_state_dict(ck["target_net_state"])
            agent.opt.load_state_dict(ck["opt_state"]); agent.global_step = ck["global_step"]
            log = ck["log"]; start = ck["episode"] + 1
            print(f"[{job}] resume tu episode {start}")
        except (RuntimeError, KeyError) as e:
            print(f"[{job}] checkpoint khong khop -> train lai ({e})")

    def save_policy(path):
        torch.save({"model_state": agent.q_net.state_dict(), "state_dim": env.state_dim,
                    "action_dim": env.action_dim, "label_mode": a.label_mode}, path)

    t0 = time.time()
    for ep in range(start, a.n_episodes):
        s = env.reset()
        ep_r, steps_done = 0.0, 0
        for _ in range(min(steps, env.n)):
            act = agent.select_action(s)
            s2, r, done, _ = env.step(act)
            agent.store(s, act, r, s2, done)
            agent.train_step()
            s, ep_r, steps_done = s2, ep_r + r, steps_done + 1
            if done:
                break
        sm = env.episode_summary()
        log["episode_rewards"].append(ep_r)
        log["episode_summaries"].append({k: v for k, v in sm.items() if k != "by_label"})
        msg = (f"[{job}] ep {ep+1}/{a.n_episodes} reward={ep_r:.1f} eps={agent.epsilon():.2f} "
               f"loss={sm['loss_rate']*100:.1f}% delay={sm['avg_delay_ms']:.1f}ms ({(time.time()-t0)/60:.1f} phut)")
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.n_episodes:
            v = validate(env, agent, val_starts, steps)
            v["episode"] = ep + 1
            log["val_history"].append(v)
            if v["reward"] > log["best_val_reward"]:
                log["best_val_reward"], log["best_episode"] = v["reward"], ep + 1
                save_policy(out_dir / f"{job}_qnet.pt")
                msg += "  *best*"
            msg += (f"\n    [val] reward={v['reward']:.1f} loss={v['loss_rate']*100:.1f}% "
                    f"delay={v['avg_delay_ms']:.1f}ms thr={v['throughput_mbps']:.2f}Mbps")
        print(msg)
        torch.save({"q_net_state": agent.q_net.state_dict(), "target_net_state": agent.target_net.state_dict(),
                    "opt_state": agent.opt.state_dict(), "global_step": agent.global_step,
                    "episode": ep, "log": log}, ck_path)

    save_policy(out_dir / f"{job}_last_qnet.pt")
    (out_dir / f"{job}_v2_log.json").write_text(json.dumps(log, indent=1))
    plot_convergence(log, out_dir / f"{job}_convergence.png")
    print(f"\n[{job}] XONG. tot nhat o episode {log['best_episode']} (val reward {log['best_val_reward']:.1f}); "
          f"chinh sach tot nhat -> {out_dir / (job + '_qnet.pt')}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["ns3", "ukm"], required=True)
    ap.add_argument("--label_mode", choices=["none", "external"], required=True)
    ap.add_argument("--n_episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=None, help="so goi moi episode (mac dinh: ns3 5000, ukm 6000)")
    ap.add_argument("--eps_decay_steps", type=int, default=150000)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--val_k", type=int, default=6)
    ap.add_argument("--train_seed", type=int, nargs="+", default=[0])
    ap.add_argument("--d_max_ms", type=float, default=None)
    ap.add_argument("--rate_mbps", type=float, default=None)
    ap.add_argument("--cong_frac", type=float, default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out_dir", default=None)
    a, _ = ap.parse_known_args()
    for sd in a.train_seed:
        train(a, sd)
