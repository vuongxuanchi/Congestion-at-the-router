"""
Tinh nhan FRCNC THAT (khong phai nhan goc/threshold-rule) cho tung cua so
(flow, 1s) trong file vet DropTail dung cho Thuc nghiem 3
(out_ip_header_droptail_seed{S}_run{R}.csv), de nhung vao trang thai
DN-dCoD dung cong thuc (3.47).

Quy trinh:
  1) Doc su kien TX cua vet DropTail (co day du 15 dac trung goi tin - xem
     SEQ_FEATURE_COLS - khac voi ENQUEUE/DEQUEUE khong co payload_size).
  2) Nhom theo luong (src_ip,dst_ip,src_port,dst_port) va cua so 1 giay
     (window = floor(Timestamp)), dung dinh dang CHUOI GOI TIN (Bo A,
     seq_len=48) - DUNG kieu du lieu FRCNC that su nhan vao, KHONG phai 8
     dac trung thong ke cua Bo B (ns3_window_6.csv chi de tham khao nhan
     GOC/threshold-rule, khong the dua truc tiep vao FRCNC).
  3) Nap checkpoint FRCNC (WCNN2WavKAN) da huan luyen, chay suy luan, lay
     nhan du doan (Mice/Sheep/Elephant) cho tung cua so.
  4) Luu bang tra cuu (src_ip,dst_ip,src_port,dst_port,window_start_s) ->
     nhan du doan, dung cho set_label_provider() trong dcod_env.py.

Usage:
  python3 compute_frcnc_labels.py --seed 10 --run 1 \
      --frcnc_ckpt /path/results/ns3-6/frcnc/frcnc_ns3_agents50_best.pt
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()  # ep Python quet lai thu muc DATA_DIR - tranh loi
                                 # ModuleNotFoundError khi file .py duoc upload len Drive
                                 # SAU KHI phien Colab da tung quet thu muc nay 1 lan

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 48
WINDOW_SECONDS = 1.0
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]

SEQ_FEATURE_COLS = [
    "payload_size", "tcp_ack_flag", "inter_arrival_ms", "ttl", "tos",
    "fragment_offset", "MF", "DF", "is_tcp", "is_udp",
    "time_in_window_s", "packet_index_in_window", "packet_position_ratio",
    "cumulative_bytes", "rolling_mean_payload",
]

# ============================== MODEL (dung y het WCNN hien tai) ============


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


class WCNN2WavKAN(_ConvBackbone):
    def __init__(self, in_features, num_classes=3):
        super().__init__(in_features, num_classes, lambda c: nn.ReLU(), WaveletKANActivation)


class WCNN2WavKANLegacy(_ConvBackbone):
    def __init__(self, in_features, num_classes=3):
        super().__init__(in_features, num_classes, lambda c: nn.ReLU(), WaveletKANActivationLegacy)


def load_frcnc(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    for cls in [WCNN2WavKAN, WCNN2WavKANLegacy]:
        model = cls(in_features=len(SEQ_FEATURE_COLS)).to(DEVICE)
        try:
            model.load_state_dict(ck["model_state"])
            model.eval()
            if cls is WCNN2WavKANLegacy:
                print(f"[frcnc] CANH BAO: nap {ckpt_path} bang KIEN TRUC CU "
                      f"(Wav-KAN 1 thanh phan/kenh) - day KHONG PHAI phien ban WCNN da toi uu "
                      f"dung trong Thuc nghiem 1/2 (Wav-KAN 3 thanh phan). Nhan FRCNC tinh ra o "
                      f"day se kem chinh xac hon model moi nhat. Neu da chay lai "
                      f"train_frcnc.py (ban da nang cap da-seed), tro --frcnc_ckpt toi "
                      f"'.../frcnc_ns3_agents50_seed0_best.pt' (co hau to _seed0) de dung ban moi.")
            else:
                print(f"[frcnc] loaded {ckpt_path} ({cls.__name__} - kien truc hien tai)")
            return model
        except RuntimeError:
            continue
    raise RuntimeError(f"Khong nap duoc {ckpt_path} voi bat ky kien truc WCNN nao da biet")


def load_scaler(seed=42):
    """Uu tien nap scaler DA FIT SAN tu cache cua train_ns3_multiseed.py
    (dung nhat quan voi luc train FRCNC) - neu khong tim thay, canh bao va
    fit tam tren chinh du lieu AQM nay (co the lech phan bo, chi la du
    phong)."""
    cache_path = Path(f"{DATA_DIR}/results/ns3-6/data_cache/ns3_split_seed{seed}.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"[scaler] nap tu cache huan luyen FRCNC: {cache_path}")
        return d["scaler_mean"], d["scaler_scale"]
    print(f"[scaler] KHONG tim thay cache {cache_path} -> se fit tam scaler tren "
          f"chinh du lieu AQM nay (CANH BAO: co the lech phan bo so voi luc train FRCNC).")
    return None, None


# ============================== BUILD SEQUENCES TU TX EVENTS =================


def build_windows_from_tx(droptail_csv):
    """Doc su kien TX, dung 15 dac trung Bo A, nhom theo (luong, cua so 1s).
    Tra ve: X (n_windows, SEQ_LEN, 15), flow_keys (list tuple (src_ip,
    dst_ip,src_port,dst_port,window_start_s))."""
    print(f"[data] reading TX events tu {droptail_csv} ...")
    usecols = ["Timestamp", "event_type", "src_ip", "dst_ip", "src_port", "dst_port",
               "protocol", "tos", "ttl", "fragment_offset", "MF", "DF",
               "payload_size", "tcp_ack_flag"]
    df = pd.read_csv(droptail_csv, usecols=usecols, low_memory=False)
    tx = df[df["event_type"] == "TX"].copy()
    del df

    tx["window_start_s"] = np.floor(tx["Timestamp"] / WINDOW_SECONDS) * WINDOW_SECONDS
    tx["is_tcp"] = (tx["protocol"] == 6).astype(np.float32)
    tx["is_udp"] = (tx["protocol"] == 17).astype(np.float32)
    tx.sort_values(["src_ip", "dst_ip", "src_port", "dst_port", "window_start_s", "Timestamp"],
                    inplace=True)

    groups = tx.groupby(["src_ip", "dst_ip", "src_port", "dst_port", "window_start_s"], sort=False)
    n = groups.ngroups
    X = np.zeros((n, SEQ_LEN, len(SEQ_FEATURE_COLS)), dtype=np.float32)
    flow_keys = []
    idx = 0
    for key, g in groups:
        g = g.reset_index(drop=True)
        t0 = float(g["Timestamp"].iloc[0])
        inter_arrival = g["Timestamp"].diff().fillna(0).to_numpy() * 1000.0
        time_in_window = (g["Timestamp"].to_numpy() - g["window_start_s"].iloc[0])
        pkt_idx = np.arange(len(g), dtype=np.float32)
        pkt_ratio = pkt_idx / max(1, len(g) - 1) if len(g) > 1 else pkt_idx
        cum_bytes = g["payload_size"].cumsum().to_numpy()
        roll_mean = g["payload_size"].expanding().mean().to_numpy()

        feat = np.stack([
            g["payload_size"].to_numpy(dtype=np.float32),
            g["tcp_ack_flag"].to_numpy(dtype=np.float32),
            inter_arrival.astype(np.float32),
            g["ttl"].to_numpy(dtype=np.float32),
            g["tos"].to_numpy(dtype=np.float32),
            g["fragment_offset"].to_numpy(dtype=np.float32),
            g["MF"].to_numpy(dtype=np.float32),
            g["DF"].to_numpy(dtype=np.float32),
            g["is_tcp"].to_numpy(dtype=np.float32),
            g["is_udp"].to_numpy(dtype=np.float32),
            time_in_window.astype(np.float32),
            pkt_idx,
            pkt_ratio.astype(np.float32),
            cum_bytes.astype(np.float32),
            roll_mean.astype(np.float32),
        ], axis=1)

        L = min(len(feat), SEQ_LEN)
        X[idx, :L, :] = feat[:L]
        flow_keys.append(key)
        idx += 1
        if idx % 20000 == 0:
            print(f"    ... da xu ly {idx} cua so")

    print(f"[data] tong cong {n} cua so (luong x 1s)")
    return X, flow_keys


def predict_labels(model, X, scaler_mean, scaler_scale, batch_size=512):
    shp = X.shape
    flat = X.reshape(-1, shp[2])
    nonzero_mask = np.any(flat != 0, axis=1)

    if scaler_mean is None:
        # fallback: fit tam scaler tren chinh du lieu nay (chi dung khi
        # KHONG tim thay cache huan luyen FRCNC that - xem canh bao da in
        # o load_scaler()).
        scaler_mean = flat[nonzero_mask].mean(axis=0)
        scaler_scale = flat[nonzero_mask].std(axis=0)
        scaler_scale[scaler_scale < 1e-6] = 1.0

    flat_scaled = flat.copy()
    flat_scaled[nonzero_mask] = (flat[nonzero_mask] - scaler_mean) / scaler_scale
    X = flat_scaled.reshape(shp).astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
            logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--frcnc_ckpt", type=str,
                     default=f"{DATA_DIR}/results/ns3-6/frcnc/frcnc_ns3_agents50_best.pt")
    ap.add_argument("--out_csv", type=str, default=None)
    cli_args, _unknown = ap.parse_known_args()

    droptail_csv = f"{DATA_DIR}/out_ip_header_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"
    out_csv = cli_args.out_csv or f"{DATA_DIR}/frcnc_labels_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"

    print(f"[checkpoint] exists={Path(cli_args.frcnc_ckpt).exists()} {cli_args.frcnc_ckpt}")
    model = load_frcnc(cli_args.frcnc_ckpt)
    scaler_mean, scaler_scale = load_scaler()

    X, flow_keys = build_windows_from_tx(droptail_csv)
    preds = predict_labels(model, X, scaler_mean, scaler_scale)

    rows = []
    for (src_ip, dst_ip, src_port, dst_port, window_start_s), p in zip(flow_keys, preds):
        rows.append(dict(src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
                          window_start_s=window_start_s, label=LABEL_NAMES[p]))
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"\n[done] phan bo nhan FRCNC du doan: {out_df['label'].value_counts().to_dict()}")
    print(f"saved -> {out_csv}")
