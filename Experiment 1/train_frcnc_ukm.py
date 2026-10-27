"""
FRCNC = Federated Averaging (FedAvg) of the WCNN traffic classifier across
N simulated router "agents", each observing a disjoint subset of flows
(no raw packet data ever leaves an agent - only model weights are
exchanged, per Chuong 3 cong thuc 3.40 - Federated Logic).

Design note / deviation from the literal Chuong 3 description: the thesis
also describes a local/global Q-Table selecting the classification action
via epsilon-greedy over a continuous state space. A tabular Q-table over
a continuous multi-dimensional state is not tractable (this is exactly
the ambiguity phan bien Nguyen Anh Linh raised: "FRCNC thuc chat lien ket
trong so WCNN, tham so chinh sach hay Q-table?"). This script implements
FRCNC as federated averaging of the WCNN's own weights (theta_i -> theta),
with the WCNN's softmax output used directly as the classification
decision - matching cong thuc (3.40) and Thuat toan 2's model-averaging
step, without the separate tabular Q-table.

Data note (UKMDDoSDN, real-world traffic - not tied to any NS-3 node
count): N_AGENTS defaults to 6 simulated router agents. The dataset is
much smaller than NS-3 (~5.6k train windows vs ~36k), so a large agent
count would leave each agent with very little data; 6 keeps each agent's
share reasonably sized while still exercising the federated-averaging
mechanics across multiple agents.

Usage:
  python3 train_frcnc_ukm.py --n_agents 6 --n_rounds 100 --local_epochs 3 \
      --init_seed 0   # loads MODELS_DIR/ukm_wcnn_seed0/best.pt as the starting
                       # global model (the WCNN you already trained with
                       # train_ukm_multiseed.py on the SAME DATA_DIR)
"""
import os
import json
import time
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

# ============================== CONFIG ======================================

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
UKM_SEQUENCE_FILE = os.environ.get("UKM_SEQUENCE_FILE", f"{DATA_DIR}/ukm_sequence_2.csv")
RESULTS_DIR = f"{DATA_DIR}/results_ukm"
TRAINING_LOG_DIR = f"{RESULTS_DIR}/training_logs"
MODELS_DIR = f"{RESULTS_DIR}/models"
DATA_CACHE_DIR = f"{RESULTS_DIR}/data_cache"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
FRCNC_DIR = f"{RESULTS_DIR}/frcnc"

for d in [RESULTS_DIR, TRAINING_LOG_DIR, MODELS_DIR, DATA_CACHE_DIR, PLOTS_DIR, FRCNC_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

SEQ_LEN = 24          # UKM window packet-sequence length (matches train_ukm_multiseed.py)
SPLIT_SEED = 42          # same fixed split as the centralized CNN/WCNN runs -> results are comparable
N_AGENTS = 6             # simulated router agents (UKM dataset is small; see note above)
N_ROUNDS = 100            # federated communication rounds
LOCAL_EPOCHS = 3         # local SGD epochs per agent per round (raised: each agent has little data)
LOCAL_LR_MAIN = 2e-4      # base LR for round 1 - smaller than centralized training (this is fine-tuning)
LOCAL_LR_WAVKAN = 1e-4
ROUND_MIN_LR_RATIO = 0.1  # by the final round, LR has decayed (cosine) to 10% of the base value
BATCH_SIZE = 128
GRAD_CLIP = 1.0
PARTITION_SEED = 0        # controls which flows go to which agent
EARLY_STOP_PATIENCE = 20  # stop if best_val_acc hasn't improved for this many rounds
                          # (a FedAvg run that has collapsed/plateaued rarely recovers;
                          # this avoids burning the remaining rounds for nothing)
BN_RECAL_BATCHES = 5      # forward-pass batches used each round to refresh BatchNorm
                          # running stats after aggregation (see recalibrate_batchnorm)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "payload_size", "tcp_ack_flag", "inter_arrival_ms", "ttl", "tos",
    "fragment_offset", "MF", "DF", "is_tcp", "is_udp",
    "time_in_window_s", "packet_index_in_window", "packet_position_ratio",
    "cumulative_bytes", "rolling_mean_payload",
]
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]

print(f"Device: {DEVICE}")

# ============================== DATA PREP ===================================
# (identical to train_ukm_multiseed.py, so the split/scaler/cache are shared)


def build_sequences(df: pd.DataFrame, seq_len: int):
    df = df.sort_values(["flow_id", "window_id", "packet_index_in_window"])
    groups = df.groupby(["flow_id", "window_id"], sort=False)
    n = groups.ngroups
    Fdim = len(FEATURE_COLS)
    X = np.zeros((n, seq_len, Fdim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    flow_ids = np.empty((n,), dtype=object)
    feat_arr_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    label_all = df["label_code"].to_numpy(dtype=np.int64)
    flow_all = df["flow_id"].to_numpy(dtype=object)
    idx = 0
    for _, g in groups.indices.items():
        g = np.asarray(g)
        L = min(len(g), seq_len)
        X[idx, :L, :] = feat_arr_all[g[:L]]
        y[idx] = label_all[g[0]]
        flow_ids[idx] = flow_all[g[0]]
        idx += 1
    return X, y, flow_ids


def prepare_ukm(seed: int = SPLIT_SEED, force: bool = False):
    cache_path = Path(DATA_CACHE_DIR) / f"ukm_split_seed{seed}_frcnc.npz"
    meta_path = Path(DATA_CACHE_DIR) / f"ukm_split_seed{seed}_frcnc_meta.json"
    if cache_path.exists() and not force:
        print(f"[data] using cached split {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data, json.loads(meta_path.read_text())

    print(f"[data] reading {UKM_SEQUENCE_FILE} ...")
    usecols = ["flow_id", "window_id", "packet_index_in_window", "label_code"] + FEATURE_COLS
    dtype_map = {c: "float32" for c in FEATURE_COLS}
    dtype_map.update({"label_code": "int8", "window_id": "int32",
                       "packet_index_in_window": "int32", "flow_id": "category"})
    df = pd.read_csv(UKM_SEQUENCE_FILE, usecols=usecols, dtype=dtype_map)
    print(f"[data] building sequences (seq_len={SEQ_LEN}) ...")
    X, y, flow_ids = build_sequences(df, SEQ_LEN)

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss1.split(X, y, groups=flow_ids))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, y[rest_idx], groups=flow_ids[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]

    scaler = StandardScaler()
    train_flat = X[train_idx].reshape(-1, X.shape[2])
    nonzero_mask = np.any(train_flat != 0, axis=1)
    scaler.fit(train_flat[nonzero_mask])

    def apply_scaler(arr):
        shp = arr.shape
        flat = arr.reshape(-1, shp[2])
        out = scaler.transform(flat).astype(np.float32).reshape(shp)
        pad_mask = ~np.any(arr != 0, axis=2)
        out[pad_mask] = 0.0
        return out

    Xn = apply_scaler(X)
    np.savez_compressed(cache_path, X=Xn, y=y, flow_ids=flow_ids,
                         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    meta = dict(seq_len=SEQ_LEN, n_features=len(FEATURE_COLS), feature_cols=FEATURE_COLS,
                n_samples=int(X.shape[0]), n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                n_test=int(len(test_idx)), split_seed=seed, label_names=LABEL_NAMES)
    meta_path.write_text(json.dumps(meta, indent=2))
    data = np.load(cache_path, allow_pickle=True)
    return data, meta


# ============================== MODEL (identical to the tuned WCNN) =========


class WaveletKANActivation(nn.Module):
    """Multi-wavelet Wav-KAN activation (must match train_ukm_multiseed.py exactly,
    or a checkpoint trained there will fail to load here with a size-mismatch error):
    N_WAVELETS learnable Mexican-Hat components per channel, summed, plus a smooth
    SiLU base term."""

    def __init__(self, num_channels: int, n_wavelets: int = 3):
        super().__init__()
        self.n_wavelets = n_wavelets
        self.scale = nn.Parameter(torch.ones(n_wavelets, num_channels))
        init_offsets = torch.linspace(-1.5, 1.5, n_wavelets).view(n_wavelets, 1)
        self.translation = nn.Parameter(init_offsets.repeat(1, num_channels).clone())
        self.wavelet_weight = nn.Parameter(torch.full((n_wavelets, num_channels), 1.0 / n_wavelets))
        self.base_weight = nn.Parameter(torch.full((num_channels,), 0.5))

    def forward(self, x):  # x: (B, C, L)
        B, C, L = x.shape
        x_exp = x.unsqueeze(1)
        s = self.scale.view(1, self.n_wavelets, C, 1).clamp(min=1e-3)
        b = self.translation.view(1, self.n_wavelets, C, 1)
        u = (x_exp - b) / s
        wavelet = (1.0 - u ** 2) * torch.exp(-0.5 * u ** 2)
        w = self.wavelet_weight.view(1, self.n_wavelets, C, 1)
        combined = (w * wavelet).sum(dim=1)
        base = self.base_weight.view(1, -1, 1) * F.silu(x)
        return combined + base


class _ConvBackbone(nn.Module):
    def __init__(self, in_features: int, num_classes: int, act1_fn, act2_fn):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = act1_fn(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = act2_fn(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class WCNN2WavKAN(_ConvBackbone):
    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__(in_features, num_classes, act1_fn=lambda c: nn.ReLU(), act2_fn=WaveletKANActivation)


def round_lr_multiplier(r, total_rounds, min_ratio=ROUND_MIN_LR_RATIO):
    """Cosine decay across FEDERATED ROUNDS (not local epochs): round 0 -> 1.0x,
    final round -> min_ratio x. No warm-up here since we start from an already
    pretrained WCNN, not from random weights."""
    progress = min(max(r / max(1, total_rounds - 1), 0.0), 1.0)
    cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cos_decay


def build_optimizer(model, lr_main, lr_wavkan):
    wavkan_params = []
    for module in model.modules():
        if isinstance(module, WaveletKANActivation):
            wavkan_params += list(module.parameters())
    wavkan_ids = {id(p) for p in wavkan_params}
    other_params = [p for p in model.parameters() if id(p) not in wavkan_ids]
    groups = [{"params": other_params, "lr": lr_main}]
    if wavkan_params:
        groups.append({"params": wavkan_params, "lr": lr_wavkan})
    return torch.optim.Adam(groups)


# ============================== METRICS =====================================


def evaluate_full(model, loader):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)


def compute_metrics(y_true, y_pred, label_names=LABEL_NAMES):
    labels = list(range(len(label_names)))
    acc = accuracy_score(y_true, y_pred)
    prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = {label_names[i]: {"precision": float(prec_pc[i]), "recall": float(rec_pc[i]),
                                   "f1": float(f1_pc[i]), "support": int(support_pc[i])}
                 for i in range(len(label_names))}
    return {"accuracy": float(acc), "precision_macro": float(prec_macro), "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro), "precision_weighted": float(prec_w), "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w), "per_class": per_class}


class SeqDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y = X[idx], y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


def run_epoch(model, loader, criterion, optimizer=None, grad_clip=GRAD_CLIP):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


# ============================== FEDERATED PARTITIONING ======================


def partition_by_flow(flow_ids, train_idx, n_agents, seed=PARTITION_SEED):
    """Assign each unique flow (among the TRAIN indices) to exactly one agent,
    so no agent ever needs another agent's raw packets - only model weights
    are exchanged afterwards."""
    train_flows = np.unique(flow_ids[train_idx])
    rng = np.random.RandomState(seed)
    rng.shuffle(train_flows)
    flow_to_agent = {f: i % n_agents for i, f in enumerate(train_flows)}
    agent_indices = [[] for _ in range(n_agents)]
    for idx in train_idx:
        a = flow_to_agent[flow_ids[idx]]
        agent_indices[a].append(idx)
    return [np.array(ix) for ix in agent_indices]


# ============================== FEDAVG =======================================


def fedavg_aggregate(local_states, sample_counts):
    """theta_global = sum_i (n_i / n) * theta_i  (cong thuc 3.40)"""
    total = sum(sample_counts)
    avg_state = copy.deepcopy(local_states[0])
    for k in avg_state.keys():
        if avg_state[k].dtype.is_floating_point:
            avg_state[k] = sum(local_states[i][k] * (sample_counts[i] / total) for i in range(len(local_states)))
        else:  # integer buffers (e.g. BatchNorm num_batches_tracked) -> just take the max
            avg_state[k] = torch.stack([local_states[i][k] for i in range(len(local_states))]).max(0).values
    return avg_state


def recalibrate_batchnorm(model, loader, num_batches=5):
    """Fix for a real instability observed on UKM/6-agent: naively averaging each
    agent's BatchNorm running_mean/running_var (as fedavg_aggregate does, same as
    every other float parameter) is NOT the correct pooled-variance formula - it
    ignores the between-client shift in means, so running_var is systematically
    UNDER-estimated. Averaged agent has been leaving BN's running_var too small
    which compounds across rounds until the network abruptly saturates/collapses.
    Cheap fix: after aggregating, run a handful of forward passes (train mode, no
    gradient) over pooled training data so BN's own momentum-based update recomputes
    running_mean/running_var directly from the current (aggregated) weights."""
    model.train()
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            model(xb.to(DEVICE))
    model.eval()


def run_frcnc(n_agents=N_AGENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
              init_seed=0, job_name=None, early_stop_patience=EARLY_STOP_PATIENCE):
    job_name = job_name or f"frcnc_ukm_agents{n_agents}"
    ckpt_path = Path(FRCNC_DIR) / f"{job_name}_latest.pt"
    log_path = Path(FRCNC_DIR) / f"{job_name}_log.json"

    data, meta = prepare_ukm(seed=SPLIT_SEED)
    X, y, flow_ids = data["X"], data["y"], data["flow_ids"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]

    agent_indices = partition_by_flow(flow_ids, train_idx, n_agents)
    print(f"[{job_name}] agent sample counts: min={min(len(a) for a in agent_indices)} "
          f"max={max(len(a) for a in agent_indices)} "
          f"mean={np.mean([len(a) for a in agent_indices]):.0f}")
    agent_loaders = [DataLoader(SeqDataset(X, y, ix), batch_size=BATCH_SIZE, shuffle=True)
                     for ix in agent_indices]
    val_loader = DataLoader(SeqDataset(X, y, val_idx), batch_size=256, shuffle=False)
    test_loader = DataLoader(SeqDataset(X, y, test_idx), batch_size=256, shuffle=False)

    # small pooled-training subset used ONLY to refresh BatchNorm running stats after
    # each round's weight aggregation (see recalibrate_batchnorm) - never used for
    # gradient updates, so this does not leak val/test information.
    rng = np.random.RandomState(PARTITION_SEED)
    recal_size = min(2000, len(train_idx))
    recal_idx = rng.choice(train_idx, size=recal_size, replace=False)
    recal_loader = DataLoader(SeqDataset(X, y, recal_idx), batch_size=BATCH_SIZE, shuffle=True)

    global_model = WCNN2WavKAN(in_features=meta["n_features"]).to(DEVICE)

    # --- start from the WCNN you already trained (centralized), if available ---
    init_ckpt = Path(MODELS_DIR) / f"ukm_wcnn_seed{init_seed}" / "best.pt"
    if init_ckpt.exists():
        ck = torch.load(init_ckpt, map_location=DEVICE)
        try:
            global_model.load_state_dict(ck["model_state"])
            print(f"[{job_name}] initialized global model from {init_ckpt}")
        except RuntimeError as e:
            print(f"[{job_name}] WARNING: could not load {init_ckpt} ({e}) -> starting from random init")
    else:
        print(f"[{job_name}] WARNING: {init_ckpt} not found -> starting FRCNC from a randomly "
              f"initialized WCNN instead of your trained one. Train ukm_wcnn_seed{init_seed} first "
              f"(train_ukm_multiseed.py) if you want FRCNC to start from it.")

    criterion = nn.CrossEntropyLoss()
    history = {"round": [], "val_loss": [], "val_acc": [], "lr_main": []}
    start_round = 0
    best_val_acc = -1.0
    rounds_since_best = 0

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE)
        try:
            global_model.load_state_dict(ck["model_state"])
            history = ck["history"]
            history.setdefault("lr_main", [])
            start_round = ck["round"] + 1
            best_val_acc = ck.get("best_val_acc", -1.0)
            print(f"[{job_name}] resuming from round {start_round}")
        except RuntimeError as e:
            print(f"[{job_name}] WARNING: checkpoint mismatch ({e}) -> starting fresh")

    for r in range(start_round, n_rounds):
        t0 = time.time()
        lr_mult = round_lr_multiplier(r, n_rounds)
        lr_main = LOCAL_LR_MAIN * lr_mult
        lr_wavkan = LOCAL_LR_WAVKAN * lr_mult

        local_states, sample_counts = [], []
        for i, loader in enumerate(agent_loaders):
            local_model = copy.deepcopy(global_model)
            optimizer = build_optimizer(local_model, lr_main, lr_wavkan)
            for _ in range(local_epochs):
                run_epoch(local_model, loader, criterion, optimizer)
            local_states.append(local_model.state_dict())
            sample_counts.append(len(loader.dataset))

        global_state = fedavg_aggregate(local_states, sample_counts)
        global_model.load_state_dict(global_state)
        recalibrate_batchnorm(global_model, recal_loader, num_batches=BN_RECAL_BATCHES)

        val_loss, val_acc = run_epoch(global_model, val_loader, criterion, None)
        history["round"].append(r)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr_main"].append(lr_main)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            rounds_since_best = 0
            torch.save({"model_state": global_model.state_dict(), "round": r, "val_acc": val_acc},
                       Path(FRCNC_DIR) / f"{job_name}_best.pt")
        else:
            rounds_since_best += 1

        torch.save({"model_state": global_model.state_dict(), "round": r, "history": history,
                    "best_val_acc": best_val_acc}, ckpt_path)
        print(f"[{job_name}] round {r+1}/{n_rounds} val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
              f"lr={lr_main:.2e} ({time.time()-t0:.1f}s){' *best*' if is_best else ''}")

        if early_stop_patience and rounds_since_best >= early_stop_patience:
            print(f"[{job_name}] EARLY STOP: no val_acc improvement for {rounds_since_best} rounds "
                  f"(best={best_val_acc:.4f}) -> stopping at round {r+1}/{n_rounds}")
            break

    best_ck = torch.load(Path(FRCNC_DIR) / f"{job_name}_best.pt", map_location=DEVICE)
    global_model.load_state_dict(best_ck["model_state"])
    y_true, y_pred = evaluate_full(global_model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    test_loss, test_acc = run_epoch(global_model, test_loader, criterion, None)
    print(f"[{job_name}] FINAL (best-round global model) test_acc={test_acc:.4f}")
    print(f"    macro P/R/F1 = {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / "
          f"{metrics['f1_macro']:.4f}")
    for cname, cm in metrics["per_class"].items():
        print(f"    {cname:10s} P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f} "
              f"support={cm['support']}")

    result = {"job_name": job_name, "n_agents": n_agents, "n_rounds": n_rounds,
              "local_epochs": local_epochs, "history": history, "best_val_acc": best_val_acc,
              "test_acc": test_acc, "test_loss": test_loss, "test_metrics": metrics}
    log_path.write_text(json.dumps(result, indent=2))
    return result


def plot_round_curve(result):
    h = result["history"]
    rounds = np.array(h["round"]) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(rounds, h["val_acc"], marker=".", color="tab:green", label="Validation accuracy")
    axes[0].set_xlabel("Federated round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"FRCNC (FedAvg, {result['n_agents']} agents) - hội tụ theo vòng - Data UKM",
                       fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(rounds, h["lr_main"], color="tab:purple")
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("LR (nhánh ReLU/conv)")
    axes[1].set_title("Lịch suy giảm LR theo vòng - Data UKM", fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_agents", type=int, default=N_AGENTS)
    ap.add_argument("--n_rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    ap.add_argument("--init_seed", type=int, default=0,
                     help="which ukm_wcnn_seed{N}/best.pt (from train_ukm_multiseed.py) to start from")
    ap.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE,
                     help="0 to disable early stopping")
    cli_args, _unknown = ap.parse_known_args()

    result = run_frcnc(n_agents=cli_args.n_agents, n_rounds=cli_args.n_rounds,
                        local_epochs=cli_args.local_epochs, init_seed=cli_args.init_seed,
                        early_stop_patience=cli_args.early_stop_patience)
    plot_round_curve(result)
