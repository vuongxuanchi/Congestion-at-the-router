"""
Thuc nghiem 3 tren DU LIEU THUC UKMDDoSDN: dCoDel / DN-dCoD (Khong nhan) /
DN-dCoD (Nhung nhan FRCNC).

UKM chi co ukm_sequence_2.csv (chuoi goi theo luong x cua so 1s), KHONG co
vet hang doi (ENQUEUE/DEQUEUE/DROP) nhu NS-3. Vi vay script nay:
  (1) DUNG LAI luong goi den tu ukm_sequence_2.csv:
        Timestamp = window_id + time_in_window_s   (giay)
      10 chunk (moi chunk phu 0-179s) duoc CHONG LEN nhau tren cung 1 truc
      thoi gian 180s (assumption: cac chunk la cac luong khac nhau cung
      thoi diem). src_ip/dst_ip/port lay tu flow_id; them tien to chunk
      ("c0|") vao IP de cac chunk khong lan khoa nhan.
      -> ghi ra out_ip_header_ukm_droptail_seed{S}_run{R}.csv (chi co TX +
         ENQUEUE, dung schema NS-3) de dung lai NGUYEN dcod_env / simulate_dcodel.
  (2) Nhan FRCNC: du doan bang checkpoint FRCNC-UKM tren dung (flow, window)
      -> bang tra cuu cung khoa voi run_dn_dcod_labeled.build_label_provider.
  (3) Chay dCoDel offline, huan luyen DN-dCoD khong nhan / nhung nhan, xuat
      vet dung schema -> compare_dcod_variants.py --seed S --run R.

QUY UOC TEN FILE: de khong de len ket qua NS-3 (seed 10), UKM dung seed=90.
  out_ip_header_{dcodel|dn_dcod_khong_nhan|dn_dcod}_seed90_run1.csv

GIOI HAN (nen neu trong luan an):
  - Day la phat lai (replay) luong thuc qua 1 hang doi mo phong, KHONG co
    phan hoi TCP; UKM la vet bat tai 1 diem nen cung khong co "DropTail"
    that de so sanh. Tai nguyen nghen co chai (--rate_mbps) la THAM SO MO
    PHONG, khong phai do tu UKM.
  - Tai trung binh UKM ~1.9 Mbps (dinh >14 Mbps/giay) nen mac dinh
    rate_mbps=2.0 de hang doi thuc su bi nen o cac dot burst.

=== BAN v3: TRUC 300 s + NHIEU LAN CHAY (seed 91) ===
  Moi "lan chay" R la mot cach xep chong khac nhau cua cung du lieu UKM tren truc T_AXIS=300 s:
    - moi chunk (phu 180 s) duoc DICH VONG TRON mot so giay nguyen ngau nhien
      (seed ngau nhien = 1000*UKM_SEED + R), roi LAP lai (tile) de phu 0-300 s;
    - tai trung binh giu nguyen (~1.9 Mbps), chi doi cach cac dot burst cua cac chunk chong nhau.
  Quy trinh (giong Thuc nghiem 3 voi NS-3): HUAN LUYEN 1 lan tren lan chay 0, roi chay GREEDY
  (khong huan luyen them) tren lan chay 1..10 -> 10 khoi cho Friedman/Wilcoxon.
  --train_chunks / --eval_chunks (vd "0,1,2,3,4" / "5,6,7,8,9") de huan luyen va danh gia tren
  cac chunk KHAC NHAU (kiem tra tong quat hoa that su; neu khong, 10 lan chay chi la 10 cach
  xep chong cua CUNG cac goi tin). Neu chia chunk, tai giam con ~nua -> can giam --rate_mbps.

Usage (Colab):
  !python3 run_ukm_dcod.py --stage train_all                 # lan chay 0: trace+labels+train x2
  !python3 run_ukm_dcod.py --stage eval --runs 1 2 3 4 5 6 7 8 9 10
  !python3 run_ukm_dcod.py --stage collect --runs 1 2 3 4 5 6 7 8 9 10   # -> aqm_wide_*_ukm.csv
  !python3 stat_tests_exp4.py --wide_csv aqm_wide_avg_delay_ms_ukm.csv --lower_better
  (cach cu: --stage all / trace / labels / dcodel / train_none / train_label cho lan chay --run)
"""
import os
import re
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()

UKM_SEQ = os.environ.get("UKM_SEQUENCE_FILE", f"{DATA_DIR}/ukm_sequence_2.csv")
UKM_FRCNC_CKPT = f"{DATA_DIR}/results_ukm/frcnc/frcnc_ukm_agents10_best.pt"
UKM_CACHE_DIR = f"{DATA_DIR}/results_ukm/data_cache"
RESULTS_DIR = f"{DATA_DIR}/results_ukm/dn_dcod_v3_300s"   # v3 (300 s, nhieu lan chay); v2: huan luyen theo doan nghen + reward D_MAX lon (khong de len ket qua v1)
D_MAX_UKM_MS = 1000.0     # v1 dung 200ms -> reward bao hoa khi tre >200ms (UKM tre toi >1000ms)
CONGESTED_FRAC = 0.7      # ty le episode bat dau tai doan NGHEN (con lai: doan ngau nhien)
SEQ_LEN = 24
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]
FEATURE_COLS = [
    "payload_size", "tcp_ack_flag", "inter_arrival_ms", "ttl", "tos",
    "fragment_offset", "MF", "DF", "is_tcp", "is_udp",
    "time_in_window_s", "packet_index_in_window", "packet_position_ratio",
    "cumulative_bytes", "rolling_mean_payload",
]
UKM_SEED, UKM_RUN = 91, 0     # v3: seed 91 de khong de len file 180 s cu (seed 90)
T_AXIS = 300.0                # do dai truc thoi gian (s)
CHUNKS = None                 # None = tat ca chunk; hoac tap chi so chunk (dung cho train/eval chia chunk)
FLOW_RE = re.compile(r"^(chunk\d+)_(.+?)__(.+)_([A-Za-z0-9]+)$")


def trace_csv():
    return f"{DATA_DIR}/out_ip_header_ukm_droptail_seed{UKM_SEED}_run{UKM_RUN}.csv"


def labels_csv():
    return f"{DATA_DIR}/frcnc_labels_ukm_seed{UKM_SEED}_run{UKM_RUN}.csv"


def _split_endpoint(s):
    """'192.168.10.10:42586.0' -> ('192.168.10.10', 42586.0); 'ip' -> ('ip', -1)."""
    if ":" in s:
        ip, port = s.rsplit(":", 1)
        return ip, float(port)
    return s, -1.0


def load_ukm_packets():
    usecols = ["flow_id", "window_id", "time_in_window_s", "packet_index_in_window",
               "label_code"] + FEATURE_COLS
    usecols = list(dict.fromkeys(usecols))
    df = pd.read_csv(UKM_SEQ, usecols=usecols)
    flows = pd.Series(df["flow_id"].unique())
    parsed = {}
    for f in flows:
        m = FLOW_RE.match(f)
        if not m:
            raise ValueError(f"flow_id khong parse duoc: {f}")
        chunk, a, b, _proto = m.groups()
        sip, sport = _split_endpoint(a)
        dip, dport = _split_endpoint(b)
        parsed[f] = (f"{chunk[0]}{chunk[5:]}|{sip}", f"{chunk[0]}{chunk[5:]}|{dip}", sport, dport)
    df["src_ip"] = df["flow_id"].map(lambda f: parsed[f][0])
    df["dst_ip"] = df["flow_id"].map(lambda f: parsed[f][1])
    df["src_port"] = df["flow_id"].map(lambda f: parsed[f][2])
    df["dst_port"] = df["flow_id"].map(lambda f: parsed[f][3])
    df = apply_layout(df)
    df["Timestamp"] = df["window_id"].astype(float) + df["time_in_window_s"].astype(float)
    return df


def apply_layout(df):
    """Xep chong cac chunk len truc T_AXIS giay cho lan chay UKM_RUN: dich vong tron moi chunk
    ngau nhien (so giay nguyen) roi lap lai de phu T_AXIS. Dac trung ben trong cua so khong doi."""
    cid = df["flow_id"].str.extract(r"^chunk(\d+)_")[0].astype(int)
    if CHUNKS is not None:
        keep = cid.isin(list(CHUNKS)).to_numpy()
        df, cid = df[keep].copy(), cid[keep]
    L = int(df["window_id"].max()) + 1
    rng = np.random.default_rng(1000 * UKM_SEED + UKM_RUN)
    shift = {c: int(rng.integers(0, L)) for c in sorted(cid.unique())}
    w = df["window_id"].to_numpy(int)
    wp = (w - cid.map(shift).to_numpy()) % L
    parts = []
    k = 0
    while k * L < T_AXIS:
        d = df.copy()
        d["window_id"] = wp + k * L
        parts.append(d[d["window_id"] < T_AXIS])
        k += 1
    out = pd.concat(parts, ignore_index=True)
    return out


# ------------------------------------------------------------------ (1) TRACE
def stage_trace():
    df = load_ukm_packets().sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    proto = np.where(df["is_tcp"] > 0, 6, np.where(df["is_udp"] > 0, 17, 1))
    base = dict(
        Timestamp=df["Timestamp"].to_numpy(), protocol=proto,
        tos=df["tos"], ttl=df["ttl"], identification=0,
        fragment_offset=df["fragment_offset"], MF=df["MF"], DF=df["DF"],
        payload_size=df["payload_size"], tcp_ack_flag=df["tcp_ack_flag"], tcp_seq=0,
        src_ip=df["src_ip"], dst_ip=df["dst_ip"],
        src_port=df["src_port"], dst_port=df["dst_port"],
    )
    tx = pd.DataFrame(base)
    tx["event_type"] = "TX"
    tx["direction"] = ""
    enq = pd.DataFrame(base)
    enq["event_type"] = "ENQUEUE"
    enq["direction"] = 0
    out = pd.concat([tx, enq], ignore_index=True).sort_values(
        ["Timestamp", "event_type"], kind="mergesort")
    for c in ["queue_size", "d_queue_ms", "rtt_context", "rtt_ms"]:
        out[c] = ""
    out.to_csv(trace_csv(), index=False)
    dur = df["Timestamp"].max() - df["Timestamp"].min()
    mbps = df["payload_size"].sum() * 8 / 1e6 / max(dur, 1e-6)
    peak = (df.groupby(df["Timestamp"].astype(int))["payload_size"].sum() * 8 / 1e6).max()
    print(f"[trace] {len(df)} goi, {dur:.1f}s, tai TB={mbps:.2f} Mbps, dinh 1s={peak:.2f} Mbps")
    print(f"[trace] -> {trace_csv()}")


# ------------------------------------------------------------------ (2) LABELS
def stage_labels(source="frcnc", ckpt=UKM_FRCNC_CKPT):
    df = load_ukm_packets().sort_values(["flow_id", "window_id", "packet_index_in_window"])
    groups = df.groupby(["flow_id", "window_id"], sort=False)
    n = groups.ngroups
    X = np.zeros((n, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float32)
    feat = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    keys, truth = [], np.zeros(n, dtype=np.int64)
    for i, (_, idx) in enumerate(groups.indices.items()):
        idx = np.asarray(idx)
        L = min(len(idx), SEQ_LEN)
        X[i, :L] = feat[idx[:L]]
        r = df.iloc[idx[0]]
        keys.append((r["src_ip"], r["dst_ip"], r["src_port"], r["dst_port"], float(r["window_id"])))
        truth[i] = int(r["label_code"])

    if source == "truth":
        preds = truth
        print("[labels] DUNG NHAN THAT cua UKM (can tren ly thuyet, KHONG phai FRCNC)")
    else:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Khong thay checkpoint FRCNC-UKM: {ckpt}\n"
                                    f"Truyen --frcnc_ckpt <duong dan> (hoac --label_source truth de thu).")
        import compute_frcnc_labels as cfl
        model = cfl.load_frcnc(ckpt)
        mean = scale = None
        for p in sorted(Path(UKM_CACHE_DIR).glob("*.npz")):
            d = np.load(p)
            if "scaler_mean" in d.files:
                mean, scale = d["scaler_mean"], d["scaler_scale"]
                print(f"[scaler] nap tu {p}")
                break
        if mean is None:
            print("[scaler] CANH BAO: khong thay cache scaler UKM -> fit tam tren du lieu nay")
        preds = cfl.predict_labels(model, X, mean, scale)
        acc = float((preds == truth).mean())
        print(f"[labels] FRCNC-UKM khop nhan that tren {n} cua so: {acc*100:.2f}%")

    rows = [dict(src_ip=k[0], dst_ip=k[1], src_port=k[2], dst_port=k[3],
                 window_start_s=k[4], label=LABEL_NAMES[p]) for k, p in zip(keys, preds)]
    out = pd.DataFrame(rows)
    out.to_csv(labels_csv(), index=False)
    print(f"[labels] phan bo: {out['label'].value_counts().to_dict()} -> {labels_csv()}")


# ------------------------------------------------------------------ (3) dCoDel
def stage_dcodel(rate_mbps):
    import simulate_dcodel as sd
    sd.BOTTLENECK_RATE_MBPS = rate_mbps
    out = f"{DATA_DIR}/out_ip_header_dcodel_seed{UKM_SEED}_run{UKM_RUN}.csv"
    sd.run_dcodel_simulation(trace_csv(), out)


# ------------------------------------------------------------------ (4) DN-dCoD
def make_env(label_mode, rate_mbps, label_fn=None):
    import dcod_env
    dcod_env.BOTTLENECK_RATE_MBPS = rate_mbps
    dcod_env.D_MAX_MS = D_MAX_UKM_MS
    from dcod_env import DCoDEnv

    class UKMEnv(DCoDEnv):
        """DCoDEnv + (a) episode huan luyen = 1 DOAN NGAU NHIEN (de khong luon
        hoc tren 5000 goi dau), (b) xuat greedy O(n) (ban goc so sanh dict
        trong list -> qua cham voi 95k goi)."""
        seg_len = None
        cong_starts = None

        def set_congested_starts(self, rate_mbps):
            """Cac chi so goi nam trong giay co tai chao > 70% dung luong nut that."""
            ts = self.arrivals["Timestamp"].to_numpy()
            sec = np.floor(ts).astype(int)
            load = np.bincount(sec, weights=self.arrivals["payload_size"].to_numpy(float)) * 8 / 1e6
            cong_sec = load > 0.7 * rate_mbps
            self.cong_starts = np.where(cong_sec[sec])[0]
            print(f"[env] {cong_sec.sum()}/{len(load)} giay nghen -> {len(self.cong_starts)} diem bat dau episode")

        def reset(self):
            s = super().reset()
            self.action_counts = [0, 0, 0, 0]
            if self.seg_len and self.n > self.seg_len:
                hi = self.n - self.seg_len
                if self.cong_starts is not None and len(self.cong_starts) and random.random() < CONGESTED_FRAC:
                    st = int(min(random.choice(self.cong_starts), hi))
                else:
                    st = random.randint(0, hi)
                self.i = st
                self.episode_start_s = float(self.arrivals["Timestamp"].iloc[st])
                self.episode_end_s = float(self.arrivals["Timestamp"].iloc[st + self.seg_len - 1])
                s = self._current_state()
            return s

        def run_greedy_and_export(self, agent, out_csv):
            """Dung HAM XUAT CUA dcod_env.DCoDEnv (da sua: DEQUEUE ghi dung thoi diem roi hang doi,
            co cong, xa het hang doi cuoi vet). Ban cu o day tu viet lai va con loi
            (DEQUEUE mang moc cua goi den ke tiep -> tre bi thoi phong). Chi them dem hanh dong."""
            self.seg_len = None
            counts = [0, 0, 0, 0]

            class _Counter:
                def __init__(self, a):
                    self.a = a

                def select_action(self, st, greedy=True):
                    x = self.a.select_action(st, greedy=greedy)
                    counts[x] += 1
                    return x

            summ = super().run_greedy_and_export(_Counter(agent), out_csv)
            tot = max(1, sum(counts))
            print("[env] phan bo hanh dong greedy (KEEP/DROP/T+/T-):",
                  [f"{100*c/tot:.1f}%" for c in counts])
            return summ

    env = UKMEnv(trace_csv(), direction=0, label_mode=label_mode, max_steps=None)
    if label_fn is not None:
        env.set_label_provider(label_fn)
    return env


def stage_train(label_mode, rate_mbps, n_episodes, steps_per_episode):
    import torch
    import train_dqn_dcod as tdd
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    tdd.RESULTS_DIR = RESULTS_DIR
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    label_fn = None
    if label_mode == "external":
        import run_dn_dcod_labeled as rl
        if not Path(labels_csv()).exists():
            raise FileNotFoundError(f"Chua co {labels_csv()} - chay --stage labels truoc.")
        label_fn = rl.build_label_provider(labels_csv())
    job = "dn_dcod_khong_nhan" if label_mode == "none" else "dn_dcod"

    env = make_env(label_mode, rate_mbps, label_fn)
    env.seg_len = steps_per_episode
    env.set_congested_starts(rate_mbps)
    agent = tdd.DQNAgent(env.state_dim, env.action_dim)
    ckpt_path = Path(RESULTS_DIR) / f"{job}_checkpoint.pt"
    rewards, losses, sums, start = [], [], [], 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=tdd.DEVICE)
        try:
            agent.q_net.load_state_dict(ck["q_net_state"])
            agent.target_net.load_state_dict(ck["target_net_state"])
            agent.opt.load_state_dict(ck["opt_state"])
            agent.global_step = ck["global_step"]
            rewards, losses, sums = ck["episode_rewards"], ck["episode_losses_avg"], ck["episode_summaries"]
            start = ck["episode"] + 1
            print(f"[{job}] resume tu episode {start}")
        except (RuntimeError, KeyError) as e:
            print(f"[{job}] checkpoint khong khop -> train lai ({e})")

    for ep in range(start, n_episodes):
        s = env.reset()
        ep_r, ep_l, steps = 0.0, [], 0
        for _ in range(min(steps_per_episode, env.n)):
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            agent.store(s, a, r, s2, done)
            l = agent.train_step()
            if l is not None:
                ep_l.append(l)
            s, ep_r, steps = s2, ep_r + r, steps + 1
            if done:
                break
        sm = env.episode_summary()
        rewards.append(ep_r); losses.append(float(np.mean(ep_l)) if ep_l else float("nan")); sums.append(sm)
        print(f"[{job}] ep {ep+1}/{n_episodes} reward={ep_r:.1f} eps={agent.epsilon():.2f} "
              f"delay={sm['avg_delay_ms']:.2f}ms loss={sm['loss_rate']*100:.2f}% "
              f"thr={sm['throughput_mbps']:.2f}Mbps T={sm['final_T_target_ms']:.1f}ms")
        if label_mode == "external" and ((ep + 1) % 10 == 0 or ep + 1 == n_episodes):
            for lb, v in sm["by_label"].items():
                if v["n_enqueue"]:
                    print(f"    [{lb:8s}] {v['share_of_traffic']*100:.1f}% loss={v['loss_rate']*100:.2f}% "
                          f"delay={v['avg_delay_ms']:.2f}ms")
        torch.save({"q_net_state": agent.q_net.state_dict(), "target_net_state": agent.target_net.state_dict(),
                    "opt_state": agent.opt.state_dict(), "global_step": agent.global_step, "episode": ep,
                    "episode_rewards": rewards, "episode_losses_avg": losses, "episode_summaries": sums},
                   ckpt_path)

    result = dict(job_name=job, label_mode=label_mode, n_episodes=n_episodes, episode_rewards=rewards,
                  episode_losses=losses, episode_summaries=sums, final_summary=sums[-1] if sums else None)
    (Path(RESULTS_DIR) / f"{job}_log.json").write_text(json.dumps(result, indent=2))
    torch.save({"model_state": agent.q_net.state_dict(), "state_dim": env.state_dim,
                "action_dim": env.action_dim, "label_mode": label_mode},
               Path(RESULTS_DIR) / f"{job}_qnet.pt")

    out = f"{DATA_DIR}/out_ip_header_{job}_seed{UKM_SEED}_run{UKM_RUN}.csv"
    fs = env.run_greedy_and_export(agent, out)
    print(f"\n[greedy eval {job}]", {k: v for k, v in fs.items() if k != "by_label"})
    if label_mode == "external":
        for lb, v in fs["by_label"].items():
            if v["n_enqueue"]:
                print(f"    [{lb:8s}] {v['share_of_traffic']*100:.1f}% loss={v['loss_rate']*100:.2f}% "
                      f"delay={v['avg_delay_ms']:.2f}ms (n={v['n_enqueue']}, drop={v['n_drop']})")


def _load_policy(job, label_mode):
    import torch
    import train_dqn_dcod as tdd
    ck = torch.load(Path(RESULTS_DIR) / f"{job}_qnet.pt", map_location=tdd.DEVICE)
    agent = tdd.DQNAgent(ck["state_dim"], ck["action_dim"])
    agent.q_net.load_state_dict(ck["model_state"]); agent.q_net.eval()
    assert ck["state_dim"] == (4 if label_mode == "none" else 7), f"{job}: state_dim lech label_mode"
    return agent


def stage_eval(runs, rate_mbps, label_source, frcnc_ckpt):
    """Chay GREEDY 2 chinh sach da huan luyen (lan chay 0) + dCoDel tren cac lan chay `runs`.
    Buoc da co ket qua se duoc bo qua (chay lai duoc sau khi Colab ngat)."""
    global UKM_RUN
    import gc
    import run_dn_dcod_labeled as rl
    agent_n = _load_policy("dn_dcod_khong_nhan", "none")
    agent_l = _load_policy("dn_dcod", "external")
    for r in runs:
        UKM_RUN = r
        print(f"\n===== UKM lan chay {r} (seed {UKM_SEED}, truc {T_AXIS:.0f}s) =====")
        if not Path(trace_csv()).exists():
            stage_trace()
        if not Path(labels_csv()).exists():
            stage_labels(label_source, frcnc_ckpt)
        out_d = f"{DATA_DIR}/out_ip_header_dcodel_seed{UKM_SEED}_run{r}.csv"
        if not Path(out_d).exists():
            stage_dcodel(rate_mbps)
        for job, agent, mode in (("dn_dcod_khong_nhan", agent_n, "none"), ("dn_dcod", agent_l, "external")):
            out = f"{DATA_DIR}/out_ip_header_{job}_seed{UKM_SEED}_run{r}.csv"
            if Path(out).exists():
                print(f"[{job}] da co"); continue
            fn = rl.build_label_provider(labels_csv()) if mode == "external" else None
            env = make_env(mode, rate_mbps, fn)
            sm = env.run_greedy_and_export(agent, out)
            print(f"[{job}] ", {k: round(v, 4) for k, v in sm.items() if k != "by_label" and isinstance(v, float)})
            if fn is not None:
                tot = max(1, fn.n_hits[0] + fn.n_miss[0])
                print(f"[{job}] nhan khop {100 * fn.n_hits[0] / tot:.1f}%")
            del env; gc.collect()


def stage_collect(runs):
    """Gom 5 chi tieu (bo phan tich da va) cua 3 co che tren cac lan chay -> aqm_wide_<metric>_ukm.csv
    (hang = lan chay, cot = co che), dung duoc truc tiep voi stat_tests_exp4.py --wide_csv."""
    import analyze_aqm_comparison as aac
    aac.TRACE_DIR = DATA_DIR      # eval ghi vet vao DATA_DIR; khong de bien moi truong LUANAN_TRACE_DIR (con sot tu thi nghiem khac) lam collect doc nham thu muc
    print(f"[collect] doc vet tu {DATA_DIR}")
    rows = []
    for r in runs:
        for a in ("dcodel", "dn_dcod_khong_nhan", "dn_dcod"):
            p = aac.trace_path(a, UKM_SEED, r)
            if not Path(p).exists():
                print(f"[thieu] {p}"); continue
            m = aac.compute_metrics_one_run(p)
            rows.append(dict(aqm=a, seed=r, **{k: m[k] for k in
                        ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]}))
            print(f"[ok] {a} lan chay {r}")
    long = pd.DataFrame(rows)
    if long.empty:
        raise SystemExit("[collect] KHONG tim thay vet nao -> khong ghi de aqm_wide_*_ukm.csv (giu nguyen file cu). Kiem tra DATA_DIR/ten vet.")
    long.to_csv(f"{DATA_DIR}/aqm_runs_long_ukm.csv", index=False)
    for m in ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]:
        long.pivot(index="seed", columns="aqm", values=m).to_csv(f"{DATA_DIR}/aqm_wide_{m}_ukm.csv")
    print(f"saved -> {DATA_DIR}/aqm_wide_<metric>_ukm.csv  (n = {long['seed'].nunique()} lan chay)")


def _parse_chunks(x):
    return None if not x or x == "all" else [int(c) for c in x.split(",")]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "trace", "labels", "dcodel", "train_none", "train_label", "train_all", "eval", "collect"])
    ap.add_argument("--run", type=int, default=0, help="lan chay dung cho trace/labels/dcodel/train (0 = lan huan luyen)")
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)), help="cac lan chay cho --stage eval/collect")
    ap.add_argument("--train_chunks", default="all")
    ap.add_argument("--eval_chunks", default="all")
    ap.add_argument("--t_axis", type=float, default=300.0)
    ap.add_argument("--policy_dir", default=None, help="thu muc chua {job}_qnet.pt cho --stage eval (vd results_ukm/dn_dcod_v4)")
    ap.add_argument("--rate_mbps", type=float, default=2.0)
    ap.add_argument("--n_episodes", type=int, default=150)
    ap.add_argument("--steps_per_episode", type=int, default=6000)
    ap.add_argument("--label_source", default="frcnc", choices=["frcnc", "truth"])
    ap.add_argument("--frcnc_ckpt", default=UKM_FRCNC_CKPT)
    a, _ = ap.parse_known_args()

    st = a.stage
    if a.policy_dir:
        RESULTS_DIR = a.policy_dir
    T_AXIS = a.t_axis
    UKM_RUN = a.run
    CHUNKS = _parse_chunks(a.train_chunks) if st in ("all", "trace", "labels", "dcodel", "train_none", "train_label", "train_all") else _parse_chunks(a.eval_chunks)
    if st in ("train_all",):
        stage_trace(); stage_labels(a.label_source, a.frcnc_ckpt); stage_dcodel(a.rate_mbps)
        stage_train("none", a.rate_mbps, a.n_episodes, a.steps_per_episode)
        stage_train("external", a.rate_mbps, a.n_episodes, a.steps_per_episode)
    if st == "eval":
        stage_eval(a.runs, a.rate_mbps, a.label_source, a.frcnc_ckpt)
    if st == "collect":
        stage_collect(a.runs)
    if st in ("all", "trace"):
        stage_trace()
    if st in ("all", "labels"):
        stage_labels(a.label_source, a.frcnc_ckpt)
    if st in ("all", "dcodel"):
        stage_dcodel(a.rate_mbps)
    if st in ("all", "train_none"):
        stage_train("none", a.rate_mbps, a.n_episodes, a.steps_per_episode)
    if st in ("all", "train_label"):
        stage_train("external", a.rate_mbps, a.n_episodes, a.steps_per_episode)
