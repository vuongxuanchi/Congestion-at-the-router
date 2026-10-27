"""
CNN vs WCNN (Wav-KAN) traffic classification on UKMDDoSDN packet-sequence data.
Runs each model 10 times (different seeds), saves a checkpoint per run
(resumable if Colab disconnects), then aggregates mean +/- std and plots
accuracy/loss curves with the std band.

Paste this into a Colab cell (after mounting Drive) or run as a script:
    !python train_ukm_multiseed.py
"""
import os
import json
import time
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
# from google.colab import drive
# drive.mount('/content/drive')

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
UKM_SEQUENCE_FILE = os.environ.get("UKM_SEQUENCE_FILE", f"{DATA_DIR}/ukm_sequence_2.csv")
RESULTS_DIR = f"{DATA_DIR}/results_ukm"
TRAINING_LOG_DIR = f"{RESULTS_DIR}/training_logs"
MODELS_DIR = f"{RESULTS_DIR}/models"
DATA_CACHE_DIR = f"{RESULTS_DIR}/data_cache"
PLOTS_DIR = f"{RESULTS_DIR}/plots"

for d in [RESULTS_DIR, TRAINING_LOG_DIR, MODELS_DIR, DATA_CACHE_DIR, PLOTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

SEQ_LEN = 24          # packets per window (covers ~90th percentile of UKM windows)
SPLIT_SEED = 42        # fixed data split -> all runs compare on the SAME train/val/test
N_SEEDS = 10            # number of independent training runs per model
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3               # base LR for conv/BN/FC params
LR_WAVKAN = 5e-4        # separate (lower) LR for Wav-KAN activation params (scale/translation/weights)
WARMUP_EPOCHS = 5       # linear warm-up epochs before cosine decay kicks in
MIN_LR_RATIO = 0.05     # cosine decay floor, as a fraction of the base LR (never fully to 0)
WEIGHT_DECAY = 1e-4     # applied to conv/FC weights only (not BN, not Wav-KAN params)
GRAD_CLIP = 1.0         # max-norm gradient clipping; set to 0 to disable
EMA_DECAY = 0.999       # weight-averaging decay; the "best" checkpoint is the EMA model,
                        # not the raw one - smooths out epoch-to-epoch noise (most visible
                        # on WCNN's validation curve) without changing what the model computes
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
    cache_path = Path(DATA_CACHE_DIR) / f"ukm_split_seed{seed}.npz"
    meta_path = Path(DATA_CACHE_DIR) / f"ukm_split_seed{seed}_meta.json"
    if cache_path.exists() and not force:
        print(f"[data] using cached split {cache_path}")
        return np.load(cache_path), json.loads(meta_path.read_text())

    print(f"[data] reading {UKM_SEQUENCE_FILE} ...")
    usecols = ["flow_id", "window_id", "packet_index_in_window", "label_code"] + FEATURE_COLS
    df = pd.read_csv(UKM_SEQUENCE_FILE, usecols=usecols)

    print(f"[data] building sequences (seq_len={SEQ_LEN}) ...")
    X, y, flow_ids = build_sequences(df, SEQ_LEN)
    print(f"[data] {X.shape[0]} samples, feature dim {X.shape[2]}")

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss1.split(X, y, groups=flow_ids))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, y[rest_idx], groups=flow_ids[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]

    assert set(flow_ids[train_idx]).isdisjoint(set(flow_ids[val_idx]))
    assert set(flow_ids[train_idx]).isdisjoint(set(flow_ids[test_idx]))
    assert set(flow_ids[val_idx]).isdisjoint(set(flow_ids[test_idx]))

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
    np.savez_compressed(cache_path, X=Xn, y=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    meta = dict(seq_len=SEQ_LEN, n_features=len(FEATURE_COLS), feature_cols=FEATURE_COLS,
                n_samples=int(X.shape[0]), n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                n_test=int(len(test_idx)), split_seed=seed, label_names=LABEL_NAMES)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[data] cached split -> {cache_path}")
    print(json.dumps(meta, indent=2))
    return np.load(cache_path), meta


# ============================== MODELS ======================================


class WaveletKANActivation(nn.Module):
    """Multi-wavelet Wav-KAN activation: N_WAVELETS learnable Mexican-Hat components
    per channel (each with its own scale/translation/mixing weight), summed, plus a
    smooth SiLU base term. A single wavelet per channel can only represent one
    "bump" in the activation curve; stacking a small mixture lets each channel learn
    a richer, multi-modal nonlinearity - closer to the original KAN idea of combining
    basis functions - while staying cheap (a few hundred extra scalars per layer)."""

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
        x_exp = x.unsqueeze(1)                                  # (B, 1, C, L)
        s = self.scale.view(1, self.n_wavelets, C, 1).clamp(min=1e-3)
        b = self.translation.view(1, self.n_wavelets, C, 1)
        u = (x_exp - b) / s
        wavelet = (1.0 - u ** 2) * torch.exp(-0.5 * u ** 2)      # (B, K, C, L)
        w = self.wavelet_weight.view(1, self.n_wavelets, C, 1)
        combined = (w * wavelet).sum(dim=1)                      # (B, C, L)
        base = self.base_weight.view(1, -1, 1) * F.silu(x)
        return combined + base


class _ConvBackbone(nn.Module):
    """conv1 (32 ch) -> act1  ;  conv2 (64 ch) -> act2 ; act1/act2 chosen per model."""

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

    def forward(self, x):  # x: (B, L, F)
        x = x.transpose(1, 2)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNN2ReLU(_ConvBackbone):
    """Layer 1: ReLU (32 ch). Layer 2: ReLU (64 ch). Pure baseline."""

    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__(in_features, num_classes,
                          act1_fn=lambda c: nn.ReLU(), act2_fn=lambda c: nn.ReLU())


class WCNN2WavKAN(_ConvBackbone):
    """Layer 1: ReLU (32 ch). Layer 2: Wav-KAN (64 ch)."""

    def __init__(self, in_features: int, num_classes: int = 3):
        super().__init__(in_features, num_classes,
                          act1_fn=lambda c: nn.ReLU(), act2_fn=WaveletKANActivation)


def build_model(name: str, in_features: int, num_classes: int = 3):
    return CNN2ReLU(in_features, num_classes) if name == "cnn" else WCNN2WavKAN(in_features, num_classes)


def build_optimizer(model, lr_main=LR, lr_wavkan=LR_WAVKAN, weight_decay=WEIGHT_DECAY):
    """Separate param groups: Wav-KAN activation params get their own (lower) LR
    and no weight decay (decaying scale/translation toward 0 would distort the
    wavelet shape); BatchNorm params also get no weight decay (standard practice).
    """
    wavkan_params, decay_params, no_decay_params = [], [], []
    for module in model.modules():
        if isinstance(module, WaveletKANActivation):
            wavkan_params += list(module.parameters())
    wavkan_ids = {id(p) for p in wavkan_params}
    for name_, p in model.named_parameters():
        if id(p) in wavkan_ids:
            continue
        if "bn" in name_ or p.ndim == 1:  # BatchNorm weight/bias, Linear/Conv bias
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    groups = [
        {"params": decay_params, "lr": lr_main, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr_main, "weight_decay": 0.0},
    ]
    if wavkan_params:
        groups.append({"params": wavkan_params, "lr": lr_wavkan, "weight_decay": 0.0})
    return torch.optim.Adam(groups)


def build_scheduler(optimizer, total_epochs, warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO):
    """Linear warm-up for `warmup_epochs`, then cosine decay down to `min_lr_ratio` * base LR.
    Works per param-group (each group's own base LR is scaled by the same factor)."""
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cos_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================== TRAIN ONE RUN ================================


class SeqDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y = X[idx], y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


def evaluate_full(model, loader):
    """Run inference over a whole loader and return (y_true, y_pred) as flat numpy arrays."""
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
    """Accuracy + precision/recall/F1 (per-class, macro, weighted)."""
    labels = list(range(len(label_names)))
    acc = accuracy_score(y_true, y_pred)
    prec_pc, rec_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = {
        label_names[i]: {"precision": float(prec_pc[i]), "recall": float(rec_pc[i]),
                          "f1": float(f1_pc[i]), "support": int(support_pc[i])}
        for i in range(len(label_names))
    }
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro), "recall_macro": float(rec_macro), "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w), "recall_weighted": float(rec_w), "f1_weighted": float(f1_w),
        "per_class": per_class,
    }


def format_metrics_line(m):
    return (f"acc={m['accuracy']:.4f} | "
            f"P(macro)={m['precision_macro']:.4f} R(macro)={m['recall_macro']:.4f} F1(macro)={m['f1_macro']:.4f} | "
            f"P(weighted)={m['precision_weighted']:.4f} R(weighted)={m['recall_weighted']:.4f} "
            f"F1(weighted)={m['f1_weighted']:.4f}")


class EMA:
    """Exponential moving average ("Polyak averaging") of a model's weights.
    Updated after every optimizer step; the shadow weights are what gets saved
    as best.pt and evaluated on val/test - they track the raw model but with
    much less epoch-to-epoch noise."""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


def run_epoch(model, loader, criterion, optimizer=None, grad_clip=GRAD_CLIP, ema=None):
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
                if ema is not None:
                    ema.update(model)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


def train_one_run(model_type: str, run_seed: int, data, meta, epochs: int = EPOCHS,
                   lr_main: float = LR, lr_wavkan: float = LR_WAVKAN,
                   warmup_epochs: int = WARMUP_EPOCHS, min_lr_ratio: float = MIN_LR_RATIO,
                   weight_decay: float = WEIGHT_DECAY, grad_clip: float = GRAD_CLIP):
    """One (model_type, run_seed) training job. Resumable via a per-run checkpoint."""
    job_name = f"ukm_{model_type}_seed{run_seed}"
    run_dir = Path(MODELS_DIR) / job_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    log_path = Path(TRAINING_LOG_DIR) / f"{job_name}.json"

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    X, y = data["X"], data["y"]
    train_idx, val_idx, test_idx = data["train_idx"], data["val_idx"], data["test_idx"]
    train_loader = DataLoader(SeqDataset(X, y, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X, y, val_idx), batch_size=256, shuffle=False)
    test_loader = DataLoader(SeqDataset(X, y, test_idx), batch_size=256, shuffle=False)

    model = build_model(model_type, in_features=meta["n_features"]).to(DEVICE)
    optimizer = build_optimizer(model, lr_main=lr_main, lr_wavkan=lr_wavkan, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs,
                                 min_lr_ratio=min_lr_ratio)
    criterion = nn.CrossEntropyLoss()
    ema = EMA(model, decay=EMA_DECAY)
    ema_model = build_model(model_type, in_features=meta["n_features"]).to(DEVICE)  # eval-only mirror

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr_main": []}
    start_epoch, best_val_acc = 0, -1.0

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE)
        try:
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optim_state"])
            if "sched_state" in ck:
                scheduler.load_state_dict(ck["sched_state"])
            if "ema_state" in ck:
                ema.load_state_dict(ck["ema_state"])
            else:
                ema = EMA(model, decay=EMA_DECAY)  # old checkpoint predates EMA -> restart shadow from here
            history = ck["history"]
            history.setdefault("lr_main", [])
            start_epoch = ck["epoch"] + 1
            best_val_acc = ck.get("best_val_acc", -1.0)
            print(f"[{job_name}] resuming from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"[{job_name}] WARNING: checkpoint at {ckpt_path} does not match the "
                  f"current architecture (probably from a previous model variant) -> "
                  f"ignoring it and starting fresh.\n   ({e})")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, grad_clip=grad_clip, ema=ema)
        ema_model.load_state_dict(ema.state_dict())
        val_loss, val_acc = run_epoch(ema_model, val_loader, criterion, None)  # validate the EMA weights
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr_main"].append(current_lr)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # best.pt stores the EMA weights as "model_state" - downstream scripts
            # (FRCNC etc.) that load best.pt automatically get the smoothed model.
            torch.save({"model_state": ema.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)

        torch.save({"model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
                     "sched_state": scheduler.state_dict(), "ema_state": ema.state_dict(),
                     "epoch": epoch, "history": history, "best_val_acc": best_val_acc}, ckpt_path)
        print(f"[{job_name}] epoch {epoch+1}/{epochs} train_acc={tr_acc:.4f} "
              f"val_acc(EMA)={val_acc:.4f} lr={current_lr:.2e} ({time.time()-t0:.1f}s){' *best*' if is_best else ''}")

    best_ck = torch.load(best_path, map_location=DEVICE)
    ema_model.load_state_dict(best_ck["model_state"])
    test_loss, test_acc = run_epoch(ema_model, test_loader, criterion, None)
    y_true, y_pred = evaluate_full(ema_model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    print(f"[{job_name}] FINAL test_acc={test_acc:.4f}")
    print(f"[{job_name}] {format_metrics_line(metrics)}")
    for cname, cm in metrics["per_class"].items():
        print(f"    {cname:10s} P={cm['precision']:.4f} R={cm['recall']:.4f} "
              f"F1={cm['f1']:.4f} support={cm['support']}")

    result = {"job_name": job_name, "model_type": model_type, "run_seed": run_seed,
              "history": history, "best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss,
              "test_metrics": metrics}
    log_path.write_text(json.dumps(result, indent=2))
    return result


# ============================== AGGREGATE + PLOT =============================


def aggregate_and_plot(model_type: str, n_seeds: int = N_SEEDS):
    train_acc, val_acc, train_loss, val_loss, test_accs = [], [], [], [], []
    for s in range(n_seeds):
        log_path = Path(TRAINING_LOG_DIR) / f"ukm_{model_type}_seed{s}.json"
        d = json.loads(log_path.read_text())
        h = d["history"]
        train_acc.append(h["train_acc"])
        val_acc.append(h["val_acc"])
        train_loss.append(h["train_loss"])
        val_loss.append(h["val_loss"])
        test_accs.append(d["test_acc"])

    train_acc, val_acc = np.array(train_acc), np.array(val_acc)
    train_loss, val_loss = np.array(train_loss), np.array(val_loss)
    test_accs = np.array(test_accs)
    n_epochs = train_acc.shape[1]
    epochs = np.arange(1, n_epochs + 1)
    n_seeds_found = train_acc.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    label = model_type.upper()

    for ax, (train_arr, val_arr, title) in zip(
        axes, [(train_acc, val_acc, "Accuracy"), (train_loss, val_loss, "Loss")]
    ):
        for i in range(n_seeds_found):
            ax.plot(epochs, train_arr[i], color="tab:blue", alpha=0.12, linewidth=0.8)
            ax.plot(epochs, val_arr[i], color="tab:orange", alpha=0.12, linewidth=0.8)

        train_mean, train_std = np.nanmean(train_arr, axis=0), np.nanstd(train_arr, axis=0)
        val_mean, val_std = np.nanmean(val_arr, axis=0), np.nanstd(val_arr, axis=0)

        ax.plot(epochs, train_mean, color="tab:blue", linewidth=2.2, label="Train (TB)")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="tab:blue", alpha=0.15)
        ax.plot(epochs, val_mean, color="tab:orange", linewidth=2.2, label="Validation (TB)")
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="tab:orange", alpha=0.15)

        ax.set_title(f"{title} vs. Epochs ({label}, n={n_seeds_found} seed)", fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
        if title == "Accuracy":
            ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out_path = Path(PLOTS_DIR) / f"ukm_{model_type}_curves_{n_seeds_found}seeds.png"
    plt.savefig(out_path, dpi=130)
    plt.show()
    plt.close()

    summary = dict(model_type=model_type, n_seeds=int(n_seeds_found),
                    test_acc_mean=float(test_accs.mean()), test_acc_std=float(test_accs.std()),
                    test_acc_per_seed=test_accs.tolist(),
                    final_val_acc_mean=float(val_acc[:, -1].mean()), final_val_acc_std=float(val_acc[:, -1].std()))
    print(f"\n[{model_type}] test_acc = {summary['test_acc_mean']:.4f} +/- {summary['test_acc_std']:.4f} "
          f"(over {n_seeds_found} runs)")
    print("plot ->", out_path)
    return summary


def summarize_metrics_table(model_types, n_seeds, job_prefix="ukm_"):
    """Load each seed's test_metrics, compute mean +/- std across seeds, print a
    comparison table (Accuracy / Precision / Recall / F1, macro + weighted, plus
    per-class F1) and save it to RESULTS_DIR/ukm_metrics_summary.csv (+ per-class csv).
    """
    rows = []
    per_class_rows = []
    for model_type in model_types:
        accs, pm, rm, fm, pw, rw, fw = [], [], [], [], [], [], []
        per_class_f1 = {c: [] for c in LABEL_NAMES}
        per_class_p = {c: [] for c in LABEL_NAMES}
        per_class_r = {c: [] for c in LABEL_NAMES}
        for s in range(n_seeds):
            log_path = Path(TRAINING_LOG_DIR) / f"{job_prefix}{model_type}_seed{s}.json"
            if not log_path.exists():
                continue
            d = json.loads(log_path.read_text())
            m = d.get("test_metrics")
            if m is None:
                continue
            accs.append(m["accuracy"]); pm.append(m["precision_macro"]); rm.append(m["recall_macro"])
            fm.append(m["f1_macro"]); pw.append(m["precision_weighted"]); rw.append(m["recall_weighted"])
            fw.append(m["f1_weighted"])
            for c in LABEL_NAMES:
                per_class_f1[c].append(m["per_class"][c]["f1"])
                per_class_p[c].append(m["per_class"][c]["precision"])
                per_class_r[c].append(m["per_class"][c]["recall"])

        if not accs:
            print(f"[summary] no test_metrics found for model_type={model_type} -> skip "
                  f"(re-run training with the updated script to generate them)")
            continue

        n_found = len(accs)
        rows.append({
            "model": model_type, "n_seeds": n_found,
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "precision_macro_mean": np.mean(pm), "precision_macro_std": np.std(pm),
            "recall_macro_mean": np.mean(rm), "recall_macro_std": np.std(rm),
            "f1_macro_mean": np.mean(fm), "f1_macro_std": np.std(fm),
            "precision_weighted_mean": np.mean(pw), "precision_weighted_std": np.std(pw),
            "recall_weighted_mean": np.mean(rw), "recall_weighted_std": np.std(rw),
            "f1_weighted_mean": np.mean(fw), "f1_weighted_std": np.std(fw),
        })
        for c in LABEL_NAMES:
            per_class_rows.append({
                "model": model_type, "class": c,
                "precision_mean": np.mean(per_class_p[c]), "precision_std": np.std(per_class_p[c]),
                "recall_mean": np.mean(per_class_r[c]), "recall_std": np.std(per_class_r[c]),
                "f1_mean": np.mean(per_class_f1[c]), "f1_std": np.std(per_class_f1[c]),
            })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df_pc = pd.DataFrame(per_class_rows)

    print("\n" + "=" * 100)
    print(f"SUMMARY (mean +/- std over seeds) -- UKM")
    print("=" * 100)
    header = f"{'Model':<8}{'n':<4}{'Accuracy':<18}{'Prec(macro)':<18}{'Recall(macro)':<18}{'F1(macro)':<18}"
    print(header)
    for r in rows:
        print(f"{r['model']:<8}{r['n_seeds']:<4}"
              f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}  "
              f"{r['precision_macro_mean']:.4f}+/-{r['precision_macro_std']:.4f}  "
              f"{r['recall_macro_mean']:.4f}+/-{r['recall_macro_std']:.4f}  "
              f"{r['f1_macro_mean']:.4f}+/-{r['f1_macro_std']:.4f}")

    print(f"\n{'Model':<8}{'n':<4}{'Prec(weighted)':<18}{'Recall(weighted)':<18}{'F1(weighted)':<18}")
    for r in rows:
        print(f"{r['model']:<8}{r['n_seeds']:<4}"
              f"{r['precision_weighted_mean']:.4f}+/-{r['precision_weighted_std']:.4f}  "
              f"{r['recall_weighted_mean']:.4f}+/-{r['recall_weighted_std']:.4f}  "
              f"{r['f1_weighted_mean']:.4f}+/-{r['f1_weighted_std']:.4f}")

    print(f"\nPer-class F1 (mean +/- std):")
    print(f"{'Model':<8}{'Class':<10}{'Precision':<18}{'Recall':<18}{'F1':<18}")
    for r in per_class_rows:
        print(f"{r['model']:<8}{r['class']:<10}"
              f"{r['precision_mean']:.4f}+/-{r['precision_std']:.4f}  "
              f"{r['recall_mean']:.4f}+/-{r['recall_std']:.4f}  "
              f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}")
    print("=" * 100)

    csv_path = Path(RESULTS_DIR) / f"{job_prefix}metrics_summary.csv"
    csv_pc_path = Path(RESULTS_DIR) / f"{job_prefix}metrics_summary_per_class.csv"
    df.to_csv(csv_path, index=False)
    df_pc.to_csv(csv_pc_path, index=False)
    print(f"\nsaved -> {csv_path}\nsaved -> {csv_pc_path}")
    return df, df_pc


# ============================== MAIN =========================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS,
                     help="e.g. --n_seeds 1 for a quick trial run")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", type=str, default="cnn,wcnn",
                     help="comma-separated, e.g. --models wcnn")
    ap.add_argument("--lr_main", type=float, default=LR)
    ap.add_argument("--lr_wavkan", type=float, default=LR_WAVKAN)
    ap.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    ap.add_argument("--min_lr_ratio", type=float, default=MIN_LR_RATIO)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--grad_clip", type=float, default=GRAD_CLIP, help="0 to disable")
    # parse_known_args (not parse_args): when this script is pasted directly into a
    # Colab/Jupyter cell (instead of run as `!python script.py ...`), sys.argv contains
    # the notebook kernel's own launcher flags (e.g. "-f .../kernel-....json"), which
    # would otherwise make argparse raise "unrecognized arguments" and exit.
    cli_args, _unknown = ap.parse_known_args()
    n_seeds, epochs = cli_args.n_seeds, cli_args.epochs
    model_types = cli_args.models.split(",")

    data, meta = prepare_ukm(seed=SPLIT_SEED)

    all_summaries = {}
    for model_type in model_types:
        for run_seed in range(n_seeds):
            train_one_run(model_type, run_seed, data, meta, epochs=epochs,
                           lr_main=cli_args.lr_main, lr_wavkan=cli_args.lr_wavkan,
                           warmup_epochs=cli_args.warmup_epochs, min_lr_ratio=cli_args.min_lr_ratio,
                           weight_decay=cli_args.weight_decay, grad_clip=cli_args.grad_clip)
        all_summaries[model_type] = aggregate_and_plot(model_type, n_seeds=n_seeds)

    summarize_metrics_table(model_types, n_seeds=n_seeds)

    # optional: significance test between CNN and WCNN test accuracy across seeds
    if "cnn" in model_types and "wcnn" in model_types and n_seeds > 1:
        try:
            from scipy import stats
            cnn_accs = [json.loads((Path(TRAINING_LOG_DIR) / f"ukm_cnn_seed{s}.json").read_text())["test_acc"] for s in range(n_seeds)]
            wcnn_accs = [json.loads((Path(TRAINING_LOG_DIR) / f"ukm_wcnn_seed{s}.json").read_text())["test_acc"] for s in range(n_seeds)]
            t_stat, p_val = stats.ttest_ind(cnn_accs, wcnn_accs)
            print(f"\nCNN vs WCNN test_acc: t={t_stat:.3f}, p={p_val:.4f} "
                  f"({'khac biet co y nghia (p<0.05)' if p_val < 0.05 else 'khong co y nghia thong ke (p>=0.05)'})")
            summary_path = Path(RESULTS_DIR) / "ukm_cnn_vs_wcnn_summary.json"
            summary_path.write_text(json.dumps({
                "cnn": all_summaries["cnn"], "wcnn": all_summaries["wcnn"],
                "ttest_t": float(t_stat), "ttest_p": float(p_val),
            }, indent=2))
            print("summary ->", summary_path)
        except ImportError:
            print("scipy not available -> skipping t-test")
