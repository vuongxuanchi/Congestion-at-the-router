"""
Gop chung 2 phep kiem tra DO TONG QUAT HOA (analyze_generalization.py +
analyze_generalization_deep.py) thanh MOT lan chay duy nhat, ra 2 bieu do
lon (NS-3 -> UKM, UKM -> NS-3), moi bieu do ve ca 6 mo hinh: RF, XGBoost,
SVM (Bo B - dac trung theo cua so) va CNN, WCNN, FRCNC (Bo A - chuoi goi
tin). Dung bieu do DUONG NOI 2 DIEM (slope chart): moi mo hinh la 1 duong
noi accuracy "cung bo" -> accuracy "bo khac" - do doc cang lon nghia la
tong quat hoa cang kem, duong cang nam ngang nghia la mo hinh giu duoc
kha nang phan loai ngay ca khi gap du lieu chua tung thay.

Usage:
  python3 analyze_generalization_all.py --n_seeds 10 --epochs 15
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xgboost as xgb

# ============================== CONFIG ======================================

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")

# Bo B (dac trung theo cua so) - cho RF / XGBoost / SVM
NS3_WINDOW_FILE = os.environ.get("NS3_WINDOW_FILE", f"{DATA_DIR}/ns3_window_1.csv")
UKM_WINDOW_FILE = os.environ.get("UKM_WINDOW_FILE", f"{DATA_DIR}/ukm_window_2.csv")

# Bo A (chuoi goi tin) - cho CNN / WCNN / FRCNC
NS3_SEQUENCE_FILE = os.environ.get("NS3_SEQUENCE_FILE", f"{DATA_DIR}/ns3_sequence_1.csv")
UKM_SEQUENCE_FILE = os.environ.get("UKM_SEQUENCE_FILE", f"{DATA_DIR}/ukm_sequence_2.csv")

# Checkpoint mac dinh - SUA TRUC TIEP O DAY neu vi tri thuc te khac
DEFAULT_NS3_CNN_CKPT = f"{DATA_DIR}/results/ns3-6/models/cnn_seed0/best.pt"
DEFAULT_NS3_WCNN_CKPT = f"{DATA_DIR}/results/ns3-6/models/wcnn_seed0/best.pt"
DEFAULT_UKM_CNN_CKPT = f"{DATA_DIR}/results_ukm/models/ukm_cnn_seed0/best.pt"
DEFAULT_UKM_WCNN_CKPT = f"{DATA_DIR}/results_ukm/models/ukm_wcnn_seed0/best.pt"
DEFAULT_FRCNC_NS3_CKPT = f"{DATA_DIR}/results/ns3-6/frcnc/frcnc_ns3_agents50_best.pt"
DEFAULT_FRCNC_UKM_CKPT = f"{DATA_DIR}/results_ukm/frcnc/frcnc_ukm_agents10_best.pt"

SPLIT_SEED = 42
NS3_SEQ_LEN = 48
UKM_SEQ_LEN = 24
N_TREES = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]
LABEL_TO_CODE = {n: i for i, n in enumerate(LABEL_NAMES)}

WINDOW_FEATURE_COLS = [
    "pkt_count_window", "volume_window", "packet_size", "packet_size_cv",
    "ack_ratio", "burstiness", "peak_rate_100ms", "duty_cycle",
]
SEQ_FEATURE_COLS = [
    "payload_size", "tcp_ack_flag", "inter_arrival_ms", "ttl", "tos",
    "fragment_offset", "MF", "DF", "is_tcp", "is_udp",
    "time_in_window_s", "packet_index_in_window", "packet_position_ratio",
    "cumulative_bytes", "rolling_mean_payload",
]

MODEL_STYLE = {
    "RF":     dict(color="#1f77b4", marker="o"),
    "XGBoost": dict(color="#ff7f0e", marker="s"),
    "SVM":    dict(color="#2ca02c", marker="^"),
    "CNN":    dict(color="#9467bd", marker="D"),
    "WCNN":   dict(color="#d62728", marker="P"),
    "FRCNC":  dict(color="#17becf", marker="*"),
}

print(f"Device: {DEVICE}")

# ============================== BO B: RF / XGBOOST / SVM =====================


def load_window_file(path, name):
    print(f"[data-B] reading {path} ...")
    df = pd.read_csv(path)
    df["label_code"] = df["label"].map(LABEL_TO_CODE)
    na_before = df[WINDOW_FEATURE_COLS].isna().sum()
    if na_before.sum() > 0:
        df[WINDOW_FEATURE_COLS] = df[WINDOW_FEATURE_COLS].fillna(0.0)
    X_train = df.loc[df["split"] == "train", WINDOW_FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = df.loc[df["split"] == "train", "label_code"].to_numpy()
    X_test = df.loc[df["split"] == "test", WINDOW_FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test = df.loc[df["split"] == "test", "label_code"].to_numpy()
    X_full = df[WINDOW_FEATURE_COLS].to_numpy(dtype=np.float64)
    y_full = df["label_code"].to_numpy()
    print(f"[data-B] {name}: n_train={len(y_train)} n_test={len(y_test)} n_full={len(y_full)}")
    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                X_full=X_full, y_full=y_full)


def fit_predict_classical(model_type, seed, Xtr, ytr, Xte, Xval=None, yval=None, scaler=None):
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
        model.fit(Xtr, ytr)
        return model.predict(Xte)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(n_estimators=N_TREES, eval_metric="mlogloss", random_state=seed,
                                    num_class=len(LABEL_NAMES), objective="multi:softprob",
                                    early_stopping_rounds=20, n_jobs=-1)
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return model.predict(Xte)
    elif model_type == "svm":
        model = SVC(kernel="rbf", random_state=seed)
        model.fit(scaler.transform(Xtr), ytr)
        return model.predict(scaler.transform(Xte))
    raise ValueError(model_type)


def run_direction_classical(train_data, eval_same, eval_cross, model_types, n_seeds):
    rng = np.random.RandomState(0)
    n = len(train_data["y_train"])
    idx = rng.permutation(n)
    val_n = max(50, int(0.15 * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]
    Xtr, ytr = train_data["X_train"][tr_idx], train_data["y_train"][tr_idx]
    Xval, yval = train_data["X_train"][val_idx], train_data["y_train"][val_idx]

    out_rows = []
    for model_type in model_types:
        n_runs = 1 if model_type == "svm" else n_seeds
        scaler = StandardScaler().fit(Xtr) if model_type == "svm" else None
        same_accs, cross_accs = [], []
        for seed in range(n_runs):
            t0 = time.time()
            pred_same = fit_predict_classical(model_type, seed, Xtr, ytr, eval_same["X_test"], Xval, yval, scaler)
            pred_cross = fit_predict_classical(model_type, seed, Xtr, ytr, eval_cross["X_full"], Xval, yval, scaler)
            same_accs.append(accuracy_score(eval_same["y_test"], pred_same))
            cross_accs.append(accuracy_score(eval_cross["y_full"], pred_cross))
            print(f"  [{model_type} seed{seed}] same={same_accs[-1]:.4f} cross={cross_accs[-1]:.4f} "
                  f"({time.time()-t0:.1f}s)")
        label = {"rf": "RF", "xgboost": "XGBoost", "svm": "SVM"}[model_type]
        out_rows.append({"model": label, "same": float(np.mean(same_accs)), "cross": float(np.mean(cross_accs))})
    return out_rows


# ============================== BO A: CNN / WCNN / FRCNC =====================


def build_sequences(df, seq_len):
    df = df.sort_values(["flow_id", "window_id", "packet_index_in_window"])
    groups = df.groupby(["flow_id", "window_id"], sort=False)
    n = groups.ngroups
    Fdim = len(SEQ_FEATURE_COLS)
    X = np.zeros((n, seq_len, Fdim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    flow_ids = np.empty((n,), dtype=object)
    feat_arr_all = df[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32)
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


def prepare_dataset(sequence_file, seq_len, seed=SPLIT_SEED):
    print(f"[data-A] reading {sequence_file} ...")
    usecols = ["flow_id", "window_id", "packet_index_in_window", "label_code"] + SEQ_FEATURE_COLS
    df = pd.read_csv(sequence_file, usecols=usecols)
    X, y, flow_ids = build_sequences(df, seq_len)

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, rest_idx = next(gss1.split(X, y, groups=flow_ids))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, y[rest_idx], groups=flow_ids[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]

    scaler = StandardScaler()
    train_flat = X[train_idx].reshape(-1, X.shape[2])
    nonzero_mask = np.any(train_flat != 0, axis=1)
    scaler.fit(train_flat[nonzero_mask])
    print(f"[data-A] n={X.shape[0]} n_train={len(train_idx)} n_val={len(val_idx)} n_test={len(test_idx)}")
    return dict(X=X, y=y, flow_ids=flow_ids, train_idx=train_idx, val_idx=val_idx,
                test_idx=test_idx, scaler=scaler)


def apply_scaler(X, scaler):
    shp = X.shape
    flat = X.reshape(-1, shp[2])
    out = scaler.transform(flat).astype(np.float32).reshape(shp)
    pad_mask = ~np.any(X != 0, axis=2)
    out[pad_mask] = 0.0
    return out


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])


class WaveletKANActivation(nn.Module):
    def __init__(self, num_channels, n_wavelets=3):
        super().__init__()
        self.n_wavelets = n_wavelets
        self.scale = nn.Parameter(torch.ones(n_wavelets, num_channels))
        init_offsets = torch.linspace(-1.5, 1.5, n_wavelets).view(n_wavelets, 1)
        self.translation = nn.Parameter(init_offsets.repeat(1, num_channels).clone())
        self.wavelet_weight = nn.Parameter(torch.full((n_wavelets, num_channels), 1.0 / n_wavelets))
        self.base_weight = nn.Parameter(torch.full((num_channels,), 0.5))

    def forward(self, x):
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


class WaveletKANActivationLegacy(nn.Module):
    """Ban Wav-KAN CU (1 thanh phan/kenh) - chi de nap tuong thich nguoc."""

    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.translation = nn.Parameter(torch.zeros(num_channels))
        self.wavelet_weight = nn.Parameter(torch.ones(num_channels))
        self.base_weight = nn.Parameter(torch.full((num_channels,), 0.5))

    def forward(self, x):
        s = self.scale.view(1, -1, 1).clamp(min=1e-3)
        b = self.translation.view(1, -1, 1)
        u = (x - b) / s
        wavelet = (1.0 - u ** 2) * torch.exp(-0.5 * u ** 2)
        w = self.wavelet_weight.view(1, -1, 1)
        base = self.base_weight.view(1, -1, 1) * F.silu(x)
        return w * wavelet + base


class _ConvBackbone(nn.Module):
    """AdaptiveAvgPool1d(1) khien model khong phu thuoc do dai chuoi dau vao -
    dung thang duoc giua seq_len=48 (NS-3) va seq_len=24 (UKM)."""

    def __init__(self, in_features, num_classes, act1_fn, act2_fn):
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


class CNN2ReLU(_ConvBackbone):
    def __init__(self, in_features, num_classes=3):
        super().__init__(in_features, num_classes, lambda c: nn.ReLU(), lambda c: nn.ReLU())


class WCNN2WavKAN(_ConvBackbone):
    def __init__(self, in_features, num_classes=3):
        super().__init__(in_features, num_classes, lambda c: nn.ReLU(), WaveletKANActivation)


class WCNN2WavKANLegacy(_ConvBackbone):
    def __init__(self, in_features, num_classes=3):
        super().__init__(in_features, num_classes, lambda c: nn.ReLU(), WaveletKANActivationLegacy)


def quick_train(model_type, data, epochs=15, lr=1e-3):
    model = (CNN2ReLU if model_type == "cnn" else WCNN2WavKAN)(in_features=len(SEQ_FEATURE_COLS)).to(DEVICE)
    Xn = apply_scaler(data["X"], data["scaler"])
    train_loader = DataLoader(SeqDataset(Xn[data["train_idx"]], data["y"][data["train_idx"]]),
                               batch_size=64, shuffle=True)
    val_loader = DataLoader(SeqDataset(Xn[data["val_idx"]], data["y"][data["val_idx"]]),
                             batch_size=256, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_val, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                n += len(yb)
        val_acc = correct / n
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"    [{model_type}] epoch {ep+1}/{epochs} val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    return model


def load_checkpoint(model_type, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    candidates = [CNN2ReLU] if model_type == "cnn" else [WCNN2WavKAN, WCNN2WavKANLegacy]
    for cls in candidates:
        model = cls(in_features=len(SEQ_FEATURE_COLS)).to(DEVICE)
        try:
            model.load_state_dict(ck["model_state"])
            tag = "" if cls is not WCNN2WavKANLegacy else " (kien truc CU)"
            print(f"    loaded checkpoint{tag}: {ckpt_path}")
            return model
        except RuntimeError:
            continue
    print(f"    WARNING: checkpoint {ckpt_path} khong khop kien truc nao -> se train nhanh thay the.")
    return None


def evaluate_deep(model, X, y, scaler, batch_size=256):
    Xn = apply_scaler(X, scaler)
    loader = DataLoader(SeqDataset(Xn, y), batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return accuracy_score(trues, preds)


def run_one_deep_model(label, model_type, train_data, eval_same, eval_cross, ckpt_path, epochs):
    model = load_checkpoint(model_type, ckpt_path) if (ckpt_path and Path(ckpt_path).exists()) else None
    if model is None:
        if label == "FRCNC":
            print(f"    khong the train FRCNC thay the (can chay lai train_frcnc*.py) -> bo qua {label}")
            return None
        print(f"    no usable checkpoint for {model_type} -> training a quick model here")
        model = quick_train(model_type, train_data, epochs=epochs)
    same = evaluate_deep(model, eval_same["X"][eval_same["test_idx"]], eval_same["y"][eval_same["test_idx"]],
                          train_data["scaler"])
    cross = evaluate_deep(model, eval_cross["X"], eval_cross["y"], train_data["scaler"])
    print(f"    {label}: same={same:.4f} cross={cross:.4f}")
    return {"model": label, "same": float(same), "cross": float(cross)}


# ============================== PLOT (SLOPE CHART) ===========================


def _declutter_labels(values_with_colors, min_gap=0.035):
    """Nhan nhung diem co gia tri qua gan nhau (vi du RF va XGBoost gan nhu
    trung khit) de tranh chu de len nhau - tra ve vi tri y da dich cho tung
    nhan, giu nguyen vi tri marker that."""
    order = sorted(range(len(values_with_colors)), key=lambda i: values_with_colors[i])
    adjusted = list(values_with_colors)
    for k in range(1, len(order)):
        i, prev_i = order[k], order[k - 1]
        if adjusted[i] - adjusted[prev_i] < min_gap:
            adjusted[i] = adjusted[prev_i] + min_gap
    return adjusted


def plot_slope(rows_ns3_to_ukm, rows_ukm_to_ns3):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    x_positions = [0, 1]
    x_labels = ["Cùng bộ\n(in-domain)", "Bộ khác\n(cross-domain)"]

    for ax, rows, title in [(axes[0], rows_ns3_to_ukm, "Huấn luyện NS-3 → đánh giá UKM"),
                              (axes[1], rows_ukm_to_ns3, "Huấn luyện UKM → đánh giá NS-3")]:
        for r in rows:
            style = MODEL_STYLE.get(r["model"], dict(color="gray", marker="o"))
            ax.plot(x_positions, [r["same"], r["cross"]],
                    color=style["color"], marker=style["marker"], markersize=9,
                    linewidth=2.2, label=r["model"], zorder=3)

        # nhan gia tri: dich rieng cho tung cot (same / cross) de tranh chong chu
        same_vals = _declutter_labels([r["same"] for r in rows])
        cross_vals = _declutter_labels([r["cross"] for r in rows])
        for r, y_same, y_cross in zip(rows, same_vals, cross_vals):
            style = MODEL_STYLE.get(r["model"], dict(color="gray", marker="o"))
            ax.annotate(f"{r['same']:.3f}", (x_positions[0] - 0.05, y_same),
                        ha="right", va="center", fontsize=8, color=style["color"], fontweight="bold")
            ax.annotate(f"{r['cross']:.3f}", (x_positions[1] + 0.05, y_cross),
                        ha="left", va="center", fontsize=8, color=style["color"], fontweight="bold")

        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    plt.suptitle("Độ tổng quát hoá liên miền: RF / XGBoost / SVM / CNN / WCNN / FRCNC\n"
                 "(đường càng dốc xuống = tổng quát hoá càng kém)",
                 fontweight="bold", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_bars(rows_ns3_to_ukm, rows_ukm_to_ns3):
    """Bieu do cot ghep doi (cung bo / bo khac), giu dung mau moi mo hinh nhu
    o plot_slope (MODEL_STYLE) - cot 'cung bo' mau dam, 'bo khac' mau nhat
    hon (alpha thap hon) cua CHINH mau do, khong doi mau theo model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, rows, title in [(axes[0], rows_ns3_to_ukm, "Huấn luyện NS-3 → đánh giá UKM"),
                              (axes[1], rows_ukm_to_ns3, "Huấn luyện UKM → đánh giá NS-3")]:
        names = [r["model"] for r in rows]
        x = np.arange(len(names))
        w = 0.35
        for i, r in enumerate(rows):
            style = MODEL_STYLE.get(r["model"], dict(color="gray"))
            ax.bar(x[i] - w/2, r["same"], w, color=style["color"], alpha=1.0,
                   edgecolor="black", linewidth=0.6,
                   label="Cùng bộ" if i == 0 else None)
            ax.bar(x[i] + w/2, r["cross"], w, color=style["color"], alpha=0.45,
                   edgecolor="black", linewidth=0.6, hatch="//",
                   label="Bộ khác" if i == 0 else None)
            ax.text(x[i] - w/2, r["same"] + 0.015, f"{r['same']:.3f}", ha="center", fontsize=8)
            ax.text(x[i] + w/2, r["cross"] + 0.015, f"{r['cross']:.3f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.suptitle("Độ tổng quát hoá liên miền — dạng cột (màu theo mô hình, "
                 "gạch chéo = bộ khác)", fontweight="bold", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================== MAIN =========================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10, help="cho RF/XGBoost")
    ap.add_argument("--epochs", type=int, default=15, help="cho CNN/WCNN khi khong nap duoc checkpoint")
    ap.add_argument("--classical_models", type=str, default="rf,xgboost,svm")
    ap.add_argument("--ns3_cnn_ckpt", default=DEFAULT_NS3_CNN_CKPT)
    ap.add_argument("--ns3_wcnn_ckpt", default=DEFAULT_NS3_WCNN_CKPT)
    ap.add_argument("--ukm_cnn_ckpt", default=DEFAULT_UKM_CNN_CKPT)
    ap.add_argument("--ukm_wcnn_ckpt", default=DEFAULT_UKM_WCNN_CKPT)
    ap.add_argument("--frcnc_ns3_ckpt", default=DEFAULT_FRCNC_NS3_CKPT)
    ap.add_argument("--frcnc_ukm_ckpt", default=DEFAULT_FRCNC_UKM_CKPT)
    cli_args, _unknown = ap.parse_known_args()

    print("[checkpoint paths]")
    for label, p in [("ns3_cnn", cli_args.ns3_cnn_ckpt), ("ns3_wcnn", cli_args.ns3_wcnn_ckpt),
                      ("ukm_cnn", cli_args.ukm_cnn_ckpt), ("ukm_wcnn", cli_args.ukm_wcnn_ckpt),
                      ("frcnc_ns3", cli_args.frcnc_ns3_ckpt), ("frcnc_ukm", cli_args.frcnc_ukm_ckpt)]:
        print(f"    {label:12s} exists={Path(p).exists() if p else False!s:6s} {p}")

    # --- Bo B (window-level) cho RF/XGBoost/SVM ---
    ns3_win = load_window_file(NS3_WINDOW_FILE, "NS-3")
    ukm_win = load_window_file(UKM_WINDOW_FILE, "UKM")
    classical_models = cli_args.classical_models.split(",")

    # --- Bo A (sequence-level) cho CNN/WCNN/FRCNC ---
    ns3_seq = prepare_dataset(NS3_SEQUENCE_FILE, NS3_SEQ_LEN)
    ukm_seq = prepare_dataset(UKM_SEQUENCE_FILE, UKM_SEQ_LEN)

    print("\n=== Chiều 1: NS-3 -> UKM ===")
    print("-- RF / XGBoost / SVM --")
    rows_ns3_to_ukm = run_direction_classical(ns3_win, ns3_win, ukm_win, classical_models, cli_args.n_seeds)
    print("-- CNN --")
    r = run_one_deep_model("CNN", "cnn", ns3_seq, ns3_seq, ukm_seq, cli_args.ns3_cnn_ckpt, cli_args.epochs)
    if r: rows_ns3_to_ukm.append(r)
    print("-- WCNN --")
    r = run_one_deep_model("WCNN", "wcnn", ns3_seq, ns3_seq, ukm_seq, cli_args.ns3_wcnn_ckpt, cli_args.epochs)
    if r: rows_ns3_to_ukm.append(r)
    print("-- FRCNC --")
    r = run_one_deep_model("FRCNC", "wcnn", ns3_seq, ns3_seq, ukm_seq, cli_args.frcnc_ns3_ckpt, cli_args.epochs)
    if r: rows_ns3_to_ukm.append(r)

    print("\n=== Chiều 2: UKM -> NS-3 ===")
    print("-- RF / XGBoost / SVM --")
    rows_ukm_to_ns3 = run_direction_classical(ukm_win, ukm_win, ns3_win, classical_models, cli_args.n_seeds)
    print("-- CNN --")
    r = run_one_deep_model("CNN", "cnn", ukm_seq, ukm_seq, ns3_seq, cli_args.ukm_cnn_ckpt, cli_args.epochs)
    if r: rows_ukm_to_ns3.append(r)
    print("-- WCNN --")
    r = run_one_deep_model("WCNN", "wcnn", ukm_seq, ukm_seq, ns3_seq, cli_args.ukm_wcnn_ckpt, cli_args.epochs)
    if r: rows_ukm_to_ns3.append(r)
    print("-- FRCNC --")
    r = run_one_deep_model("FRCNC", "wcnn", ukm_seq, ukm_seq, ns3_seq, cli_args.frcnc_ukm_ckpt, cli_args.epochs)
    if r: rows_ukm_to_ns3.append(r)

    def print_table(rows, title):
        print(f"\n{'='*70}\n{title}\n{'='*70}")
        print(f"{'Model':<10}{'Acc cùng bộ':<15}{'Acc bộ khác':<15}{'Mức sụt (đ%)':<15}")
        for r in rows:
            print(f"{r['model']:<10}{r['same']:<15.4f}{r['cross']:<15.4f}{(r['same']-r['cross'])*100:<15.2f}")

    print_table(rows_ns3_to_ukm, "NS-3 -> UKM")
    print_table(rows_ukm_to_ns3, "UKM -> NS-3")

    plot_slope(rows_ns3_to_ukm, rows_ukm_to_ns3)
    plot_bars(rows_ns3_to_ukm, rows_ukm_to_ns3)
