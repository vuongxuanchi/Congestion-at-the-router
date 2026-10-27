"""
Thi nghiem khu bien (ablation) DUNG cho WCNN, thay the bang cu trong Thuc
nghiem 2 (giam he so co gian + LayerScale - KHONG khop voi kien truc thuc
te da mo ta o muc 4.2.1). Chuoi 4 buoc duoi day di dung theo trinh tu phat
trien THAT cua kien truc (moi buoc chi doi dung MOT yeu to so voi buoc
truoc):

  v1_base       Wav-KAN 1 wavelet/kenh, MOT optimizer/LR duy nhat (1e-3)
                cho toan bo tham so, KHONG warm-up/cosine, KHONG EMA.
                -> duong co so ban dau.
  v2_lr_sched   + tach optimizer: LR rieng cho nhanh Wav-KAN (5e-4, khong
                weight-decay) va nhanh con lai (1e-3) + 5 epoch warm-up +
                suy giam cosine con 5% o epoch cuoi.
                -> kiem chung gia thuyet: dung chung 1 LR cho ca tham so
                Wav-KAN (nhay hon) va tham so tich chap thuong gay bat on.
  v3_ema        + EMA (Polyak averaging, beta=0.999) - chon epoch tot nhat
                va danh gia cuoi cung tren trong so EMA thay vi trong so
                tho.
                -> kiem chung hieu qua lam muot dao dong giua cac epoch.
  v4_multiwave  + tang so thanh phan wavelet tu 1 len 3/kenh (day chinh la
                WCNN dang dung o Thuc nghiem 1 / train_ns3_multiseed.py).
                -> kiem chung nang luc bieu dien: nhieu wavelet co giup
                xap xi tot hon ranh gioi giua 3 lop luu luong khong.

Usage:
  python3 train_wcnn_ablation.py --n_seeds 10 --epochs 50
  python3 train_wcnn_ablation.py --n_seeds 2 --epochs 5 --versions v1_base,v4_multiwave  # test nhanh
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

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
NS3_SEQUENCE_FILE = os.environ.get("NS3_SEQUENCE_FILE", f"{DATA_DIR}/ns3_sequence_6.csv")
RESULTS_DIR = f"{DATA_DIR}/results/ns3-6"
TRAINING_LOG_DIR = f"{RESULTS_DIR}/training_logs"
MODELS_DIR = f"{RESULTS_DIR}/models"
DATA_CACHE_DIR = f"{RESULTS_DIR}/data_cache"
PLOTS_DIR = f"{RESULTS_DIR}/plots"
ABLATION_DIR = f"{RESULTS_DIR}/ablation"

for d in [RESULTS_DIR, TRAINING_LOG_DIR, MODELS_DIR, DATA_CACHE_DIR, PLOTS_DIR, ABLATION_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

SEQ_LEN = 48
SPLIT_SEED = 42
N_SEEDS = 10
EPOCHS = 50
BATCH_SIZE = 64
LR_MAIN = 1e-3
LR_WAVKAN = 5e-4
WARMUP_EPOCHS = 5
MIN_LR_RATIO = 0.05
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "payload_size", "tcp_ack_flag", "inter_arrival_ms", "ttl", "tos",
    "fragment_offset", "MF", "DF", "is_tcp", "is_udp",
    "time_in_window_s", "packet_index_in_window", "packet_position_ratio",
    "cumulative_bytes", "rolling_mean_payload",
]
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]

# 4 buoc khu bien - dung y het tu ngu se in ra bang/CSV
VERSIONS = {
    "v1_base":      dict(n_wavelets=1, separate_lr=False, lr_schedule=False, use_ema=False,
                          label="WCNN gốc (1 wavelet, 1 LR, không lịch, không EMA)"),
    "v2_lr_sched":  dict(n_wavelets=1, separate_lr=True,  lr_schedule=True,  use_ema=False,
                          label="+ tách LR Wav-KAN + warm-up/cosine"),
    "v3_ema":       dict(n_wavelets=1, separate_lr=True,  lr_schedule=True,  use_ema=True,
                          label="+ EMA (Polyak averaging)"),
    "v4_multiwave": dict(n_wavelets=3, separate_lr=True,  lr_schedule=True,  use_ema=True,
                          label="+ Wav-KAN đa thành phần (3 wavelet/kênh)"),
}

print(f"Device: {DEVICE}")

# ============================== DATA PREP (giong het train_ns3_multiseed.py) =


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


def prepare_ns3(seed: int = SPLIT_SEED, force: bool = False):
    cache_path = Path(DATA_CACHE_DIR) / f"ns3_split_seed{seed}.npz"
    meta_path = Path(DATA_CACHE_DIR) / f"ns3_split_seed{seed}_meta.json"
    if cache_path.exists() and not force:
        print(f"[data] using cached split {cache_path}")
        return np.load(cache_path), json.loads(meta_path.read_text())

    print(f"[data] reading {NS3_SEQUENCE_FILE} ...")
    usecols = ["flow_id", "window_id", "packet_index_in_window", "label_code"] + FEATURE_COLS
    dtype_map = {c: "float32" for c in FEATURE_COLS}
    dtype_map.update({"label_code": "int8", "window_id": "int32",
                       "packet_index_in_window": "int32", "flow_id": "category"})
    df = pd.read_csv(NS3_SEQUENCE_FILE, usecols=usecols, dtype=dtype_map)

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
    np.savez_compressed(cache_path, X=Xn, y=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    meta = dict(seq_len=SEQ_LEN, n_features=len(FEATURE_COLS), n_samples=int(X.shape[0]),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)))
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return np.load(cache_path), meta


class SeqDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X, self.y = X[idx], y[idx]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


# ============================== MODEL (n_wavelets tham so hoa) ==============


class WaveletKANActivation(nn.Module):
    """n_wavelets=1 -> dung y het ban WCNN GOC (1 thanh phan/kenh);
    n_wavelets=3 -> dung y het ban WCNN hien tai (Thuc nghiem 1)."""

    def __init__(self, num_channels: int, n_wavelets: int = 3):
        super().__init__()
        self.n_wavelets = n_wavelets
        self.scale = nn.Parameter(torch.ones(n_wavelets, num_channels))
        if n_wavelets > 1:
            init_offsets = torch.linspace(-1.5, 1.5, n_wavelets).view(n_wavelets, 1)
            self.translation = nn.Parameter(init_offsets.repeat(1, num_channels).clone())
        else:
            self.translation = nn.Parameter(torch.zeros(n_wavelets, num_channels))
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


class WCNNAblation(nn.Module):
    """ReLU o khoi tich chap thu nhat (32 kenh, khong doi giua cac phien
    ban), Wav-KAN o khoi thu hai (64 kenh, n_wavelets tuy phien ban)."""

    def __init__(self, in_features: int, n_wavelets: int, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = WaveletKANActivation(64, n_wavelets=n_wavelets)
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


def build_optimizer(model, separate_lr: bool, lr_main=LR_MAIN, lr_wavkan=LR_WAVKAN,
                     weight_decay=WEIGHT_DECAY):
    """separate_lr=False (v1): MOT nhom duy nhat, LR=lr_main cho TOAN BO
    tham so (ke ca Wav-KAN), weight_decay ap dung dong deu (tru BN, theo
    thong le chuan). separate_lr=True (v2-v4): tach nhom Wav-KAN nhu ban
    WCNN hien tai."""
    wavkan_params = []
    for module in model.modules():
        if isinstance(module, WaveletKANActivation):
            wavkan_params += list(module.parameters())
    wavkan_ids = {id(p) for p in wavkan_params}

    decay_params, no_decay_params = [], []
    for name_, p in model.named_parameters():
        if id(p) in wavkan_ids:
            continue
        if "bn" in name_ or p.ndim == 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    if not separate_lr:
        # v1: khong tach nhom - Wav-KAN dung CHUNG lr_main, CHUNG weight_decay
        # nhu cac tham so tich chap khac (mo phong dung "chua nghi den viec
        # tach LR" o phien ban dau tien).
        groups = [
            {"params": decay_params + wavkan_params, "lr": lr_main, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr_main, "weight_decay": 0.0},
        ]
    else:
        groups = [
            {"params": decay_params, "lr": lr_main, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr_main, "weight_decay": 0.0},
        ]
        if wavkan_params:
            groups.append({"params": wavkan_params, "lr": lr_wavkan, "weight_decay": 0.0})
    return torch.optim.Adam(groups)


def build_scheduler(optimizer, total_epochs, warmup_epochs=WARMUP_EPOCHS, min_lr_ratio=MIN_LR_RATIO):
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cos_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
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


# ============================== METRICS (giong het train_ns3_multiseed.py) ==


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


# ============================== TRAIN ONE (version, seed) ===================


def train_one_run(version: str, run_seed: int, data, meta, epochs: int = EPOCHS):
    cfg = VERSIONS[version]
    job_name = f"ablation_{version}_seed{run_seed}"
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

    model = WCNNAblation(in_features=meta["n_features"], n_wavelets=cfg["n_wavelets"]).to(DEVICE)
    optimizer = build_optimizer(model, separate_lr=cfg["separate_lr"])
    scheduler = build_scheduler(optimizer, total_epochs=epochs) if cfg["lr_schedule"] else None
    criterion = nn.CrossEntropyLoss()
    ema = EMA(model, decay=EMA_DECAY) if cfg["use_ema"] else None
    eval_model = WCNNAblation(in_features=meta["n_features"], n_wavelets=cfg["n_wavelets"]).to(DEVICE) \
        if cfg["use_ema"] else model

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch, best_val_acc = 0, -1.0

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE)
        try:
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optim_state"])
            if scheduler is not None and "sched_state" in ck:
                scheduler.load_state_dict(ck["sched_state"])
            if ema is not None and "ema_state" in ck:
                ema.shadow = {k: v.clone() for k, v in ck["ema_state"].items()}
            history = ck["history"]
            start_epoch = ck["epoch"] + 1
            best_val_acc = ck.get("best_val_acc", -1.0)
            print(f"[{job_name}] resuming from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"[{job_name}] WARNING: checkpoint mismatch -> starting fresh ({e})")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, ema=ema)
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            eval_model.load_state_dict(ema.state_dict())
        val_loss, val_acc = run_epoch(eval_model, val_loader, criterion, None)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({"model_state": eval_model.state_dict(), "epoch": epoch, "val_acc": val_acc}, best_path)

        save_dict = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
                     "epoch": epoch, "history": history, "best_val_acc": best_val_acc}
        if scheduler is not None:
            save_dict["sched_state"] = scheduler.state_dict()
        if ema is not None:
            save_dict["ema_state"] = ema.state_dict()
        torch.save(save_dict, ckpt_path)
        print(f"[{job_name}] epoch {epoch+1}/{epochs} train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.1f}s){' *best*' if is_best else ''}")

    best_ck = torch.load(best_path, map_location=DEVICE)
    eval_model.load_state_dict(best_ck["model_state"])
    test_loss, test_acc = run_epoch(eval_model, test_loader, criterion, None)
    y_true, y_pred = evaluate_full(eval_model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    print(f"[{job_name}] FINAL test_acc={test_acc:.4f} F1(macro)={metrics['f1_macro']:.4f}")

    result = {"job_name": job_name, "version": version, "run_seed": run_seed, "history": history,
              "best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss,
              "test_metrics": metrics}
    log_path.write_text(json.dumps(result, indent=2))
    return result


# ============================== SUMMARY ======================================


def summarize(versions, n_seeds):
    rows = []
    val_loss_stability = {}
    for version in versions:
        accs, fms, fws = [], [], []
        val_loss_max_jump = []  # do dao dong lon nhat giua 2 epoch lien tiep (phat hien sup do)
        for s in range(n_seeds):
            log_path = Path(TRAINING_LOG_DIR) / f"ablation_{version}_seed{s}.json"
            if not log_path.exists():
                continue
            d = json.loads(log_path.read_text())
            accs.append(d["test_acc"]); fms.append(d["test_metrics"]["f1_macro"])
            fws.append(d["test_metrics"]["f1_weighted"])
            vl = np.array(d["history"]["val_loss"])
            if len(vl) > 1:
                val_loss_max_jump.append(float(np.max(np.diff(vl))))
        if not accs:
            continue
        rows.append({
            "version": version, "label": VERSIONS[version]["label"], "n_seeds": len(accs),
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "f1_macro_mean": np.mean(fms), "f1_macro_std": np.std(fms),
            "f1_weighted_mean": np.mean(fws), "f1_weighted_std": np.std(fws),
            "val_loss_max_jump_mean": np.mean(val_loss_max_jump) if val_loss_max_jump else float("nan"),
        })

    if not rows:
        print("[summary] nothing to summarize")
        return None

    df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print("BANG THUC NGHIEM 2 -- KHU BIEN WCNN (mean +/- std qua cac seed)")
    print("=" * 110)
    print(f"{'Phien ban':<16}{'n':<4}{'Accuracy':<20}{'F1(macro)':<20}{'F1(weighted)':<20}{'Dao dong loss lon nhat':<22}")
    for r in rows:
        print(f"{r['version']:<16}{r['n_seeds']:<4}"
              f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}   "
              f"{r['f1_macro_mean']:.4f}+/-{r['f1_macro_std']:.4f}   "
              f"{r['f1_weighted_mean']:.4f}+/-{r['f1_weighted_std']:.4f}   "
              f"{r['val_loss_max_jump_mean']:.4f}")
    print("=" * 110)
    print("\nMo ta tung phien ban:")
    for r in rows:
        print(f"  {r['version']:<16} {r['label']}")

    csv_path = Path(ABLATION_DIR) / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved -> {csv_path}")
    return df


def plot_ablation(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(df))
    colors = ["#9e9e9e", "#64b5f6", "#4caf50", "#d62728"]

    ax = axes[0]
    ax.bar(x, df["accuracy_mean"], yerr=df["accuracy_std"], capsize=4,
           color=colors[:len(df)], edgecolor="black")
    for i, v in enumerate(df["accuracy_mean"]):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["version"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy qua từng bước khử biến", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x, df["val_loss_max_jump_mean"], color=colors[:len(df)], edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["version"], rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Dao động val_loss lớn nhất giữa 2 epoch")
    ax2.set_title("Mức độ bất ổn huấn luyện", fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


# ============================== MAIN =========================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--versions", type=str, default="v1_base,v2_lr_sched,v3_ema,v4_multiwave")
    cli_args, _unknown = ap.parse_known_args()

    versions = cli_args.versions.split(",")
    data, meta = prepare_ns3(seed=SPLIT_SEED)

    for version in versions:
        for run_seed in range(cli_args.n_seeds):
            train_one_run(version, run_seed, data, meta, epochs=cli_args.epochs)

    df = summarize(versions, cli_args.n_seeds)
    if df is not None:
        plot_ablation(df)
