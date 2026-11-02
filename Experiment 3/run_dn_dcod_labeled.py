"""
Noi bang nhan FRCNC THAT (tu compute_frcnc_labels.py) vao DCoDEnv, roi chay
huan luyen DN-dCoD (co nhan) - buoc cuoi cung cua ke hoach: DN-dCoD khong
nhan -> DN-dCoD co nhan (nhan gia -> nhan FRCNC that) -> so sanh Thuc
nghiem 3.

CHAY THEO DUNG THU TU:
  1) python3 compute_frcnc_labels.py --seed 10 --run 1 --frcnc_ckpt <duong dan best.pt>
     -> tao ra frcnc_labels_droptail_seed10_run1.csv
  2) python3 run_dn_dcod_labeled.py --seed 10 --run 1
     -> doc file tren, gan lam label_provider, train DN-dCoD (co nhan)

Usage:
  python3 run_dn_dcod_labeled.py --n_episodes 100 --steps_per_episode 5000
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()  # ep Python quet lai thu muc DATA_DIR - tranh loi
                                 # ModuleNotFoundError khi file .py duoc upload len Drive
                                 # SAU KHI phien Colab da tung quet thu muc nay 1 lan

from dcod_env import DCoDEnv
import train_dqn_dcod as tdd

WINDOW_SECONDS = 1.0


def build_label_provider(labels_csv):
    """Doc bang tra cuu (src_ip,dst_ip,src_port,dst_port,window_start_s)->label
    tu compute_frcnc_labels.py, tra ve ham fn(row) dung cho
    env.set_label_provider(). row la 1 dong trong DCoDEnv.arrivals (co
    Timestamp, src_ip, dst_ip, src_port, dst_port)."""
    df = pd.read_csv(labels_csv)
    lookup = {}
    for r in df.itertuples(index=False):
        key = (r.src_ip, r.dst_ip, float(r.src_port), float(r.dst_port), float(r.window_start_s))
        lookup[key] = r.label

    n_hits = [0]
    n_miss = [0]

    def fn(row):
        window_start_s = np.floor(float(row["Timestamp"]) / WINDOW_SECONDS) * WINDOW_SECONDS
        key = (row["src_ip"], row["dst_ip"], float(row["src_port"]), float(row["dst_port"]), window_start_s)
        label = lookup.get(key)
        if label is None:
            n_miss[0] += 1
            return "Sheep"   # fallback trung tinh neu khong tim thay (vi du goi ENQUEUE
                              # khong ghep duoc port qua TX - xem dcod_env.py::load_arrivals)
        n_hits[0] += 1
        return label

    fn.n_hits, fn.n_miss = n_hits, n_miss
    return fn


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--steps_per_episode", type=int, default=5000)
    ap.add_argument("--direction", type=int, default=0)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--labels_csv", type=str, default=None)
    ap.add_argument("--job_name", type=str, default="dn_dcod")
    cli_args, _unknown = ap.parse_known_args()

    droptail_csv = f"{DATA_DIR}/out_ip_header_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"
    labels_csv = cli_args.labels_csv or f"{DATA_DIR}/frcnc_labels_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"

    if not Path(labels_csv).exists():
        raise FileNotFoundError(
            f"Khong tim thay {labels_csv} - chay compute_frcnc_labels.py truoc:\n"
            f"  python3 compute_frcnc_labels.py --seed {cli_args.seed} --run {cli_args.run} "
            f"--frcnc_ckpt <duong dan best.pt>")

    print(f"[label] doc bang nhan FRCNC tu {labels_csv}")
    label_fn = build_label_provider(labels_csv)

    import random
    random.seed(0)
    np.random.seed(0)
    import torch
    torch.manual_seed(0)

    env = DCoDEnv(droptail_csv, direction=cli_args.direction, label_mode="external", max_steps=None)
    env.set_label_provider(label_fn)

    # tai su dung dung logic train()/plot_training_curve() cua
    # train_dqn_dcod.py, chi thay the env da co san label_provider thay vi
    # de train() tu tao env moi (khong gan nhan).
    agent = tdd.DQNAgent(env.state_dim, env.action_dim)
    Path(tdd.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(tdd.RESULTS_DIR) / f"{cli_args.job_name}_checkpoint.pt"

    episode_rewards, episode_losses_avg, episode_summaries = [], [], []
    start_episode = 0
    if ckpt_path.exists():
        import torch as _torch
        ck = _torch.load(ckpt_path, map_location=tdd.DEVICE)
        try:
            agent.q_net.load_state_dict(ck["q_net_state"])
            agent.target_net.load_state_dict(ck["target_net_state"])
            agent.opt.load_state_dict(ck["opt_state"])
            agent.global_step = ck["global_step"]
            episode_rewards = ck["episode_rewards"]
            episode_losses_avg = ck["episode_losses_avg"]
            episode_summaries = ck["episode_summaries"]
            start_episode = ck["episode"] + 1
            print(f"[{cli_args.job_name}] resuming from episode {start_episode}")
        except (RuntimeError, KeyError) as e:
            print(f"[{cli_args.job_name}] WARNING: checkpoint khong khop -> train lai tu dau ({e})")

    for ep in range(start_episode, cli_args.n_episodes):
        s = env.reset()
        ep_reward, ep_losses, n_steps = 0.0, [], 0
        max_steps = min(cli_args.steps_per_episode, env.n)
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
        print(f"[{cli_args.job_name}] episode {ep+1}/{cli_args.n_episodes} steps={n_steps} "
              f"reward={ep_reward:.2f} eps={agent.epsilon():.3f} "
              f"delay={summary['avg_delay_ms']:.2f}ms loss_rate={summary['loss_rate']*100:.2f}% "
              f"throughput={summary['throughput_mbps']:.2f}Mbps "
              f"(nhan: hit={label_fn.n_hits[0]} miss={label_fn.n_miss[0]})")
        # in chi tiet theo tung nhan moi 10 episode (va episode cuoi cung) -
        # de kiem tra mat goi tang them co TAP TRUNG dung vao nhan duoc uu
        # tien do tre (vi du Mice) hay dan deu bat ke nhan.
        if (ep + 1) % 10 == 0 or (ep + 1) == cli_args.n_episodes:
            for lbl, s_lbl in summary["by_label"].items():
                if s_lbl["n_enqueue"] > 0:
                    print(f"    [{lbl:8s}] ty le luu luong={s_lbl['share_of_traffic']*100:.1f}% "
                          f"loss_rate={s_lbl['loss_rate']*100:.2f}% "
                          f"delay={s_lbl['avg_delay_ms']:.2f}ms "
                          f"(n_enqueue={s_lbl['n_enqueue']}, n_drop={s_lbl['n_drop']})")

        import torch as _torch
        _torch.save({
            "q_net_state": agent.q_net.state_dict(), "target_net_state": agent.target_net.state_dict(),
            "opt_state": agent.opt.state_dict(), "global_step": agent.global_step, "episode": ep,
            "episode_rewards": episode_rewards, "episode_losses_avg": episode_losses_avg,
            "episode_summaries": episode_summaries,
        }, ckpt_path)

    result = dict(job_name=cli_args.job_name, label_mode="external", n_episodes=cli_args.n_episodes,
                  episode_rewards=episode_rewards, episode_losses=episode_losses_avg,
                  episode_summaries=episode_summaries,
                  final_summary=episode_summaries[-1] if episode_summaries else None)
    import json
    log_path = Path(tdd.RESULTS_DIR) / f"{cli_args.job_name}_log.json"
    log_path.write_text(json.dumps(result, indent=2))
    print(f"saved -> {log_path}")

    ckpt_final = Path(tdd.RESULTS_DIR) / f"{cli_args.job_name}_qnet.pt"
    import torch as _torch
    _torch.save({"model_state": agent.q_net.state_dict(), "state_dim": env.state_dim,
                 "action_dim": env.action_dim, "label_mode": "external"}, ckpt_final)

    tdd.plot_training_curve(result, cli_args.job_name)

    export_csv = f"{DATA_DIR}/out_ip_header_{cli_args.job_name}_seed{cli_args.seed}_run{cli_args.run}.csv"
    final_summary = env.run_greedy_and_export(agent, export_csv)
    print(f"\n[greedy eval] {final_summary}")
    print("\n[greedy eval - theo tung nhan] (day la so lieu QUAN TRONG NHAT de tra loi cau hoi:")
    print(" mat goi tang them khi co nhan co TAP TRUNG dung vao nhan uu tien do tre khong)")
    for lbl, s_lbl in final_summary["by_label"].items():
        if s_lbl["n_enqueue"] > 0:
            print(f"    [{lbl:8s}] ty le luu luong={s_lbl['share_of_traffic']*100:.1f}% "
                  f"loss_rate={s_lbl['loss_rate']*100:.2f}% "
                  f"delay={s_lbl['avg_delay_ms']:.2f}ms "
                  f"(n_enqueue={s_lbl['n_enqueue']}, n_drop={s_lbl['n_drop']})")
    print(f"[nhan] tong so lan tra cuu: hit={label_fn.n_hits[0]} miss={label_fn.n_miss[0]} "
          f"({100*label_fn.n_hits[0]/max(1,label_fn.n_hits[0]+label_fn.n_miss[0]):.1f}% khop)")
