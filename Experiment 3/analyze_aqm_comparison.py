"""
Thuc nghiem 3 - So sanh co che dieu khien hang doi (AQM), tinh truc tiep tu
cac file vet goi tin NS-3 da co san (out_ip_header_{aqm}_seed{S}_run{R}.csv),
KHONG can chay lai mo phong hay viet .cc moi cho cac co che chuan da co du
lieu: DropTail, CoDel, FQ-CoDel, PIE, CAKE.

CHUA co du lieu cho: dCoDel, DQN-dCoD, DN-dCoD (3 co che con lai do
luan an tu de xuat/cai dat - can xac nhan rieng truoc khi lam tiep).

Tinh 5 chi tieu theo dung mo ta muc 4.3.2:
  - Do tre hang doi trung binh (D_bar)      : mean(d_queue_ms) tai DEQUEUE
  - Do dai hang doi trung binh (Q_bar)      : dinh luat Little = tong thoi gian cho cua cac goi / thoi luong
  - Ty le mat goi (Packet Loss Rate)        : DROP / (ENQUEUE + DROP truoc khi vao hang doi)
  - Thong luong (Throughput)                : tong byte RX / thoi gian mo phong
  - Jain's Fairness Index                   : theo thong luong tung luong (5-tuple)

Usage:
  python3 analyze_aqm_comparison.py --n_seeds 1
  # hoac chi mot vai seed/run neu chua co du:
  python3 analyze_aqm_comparison.py --seeds 10 --runs 1
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
TRACE_DIR = os.environ.get("LUANAN_TRACE_DIR", DATA_DIR)   # thu muc chua vet out_ip_header_*.csv

# 7 co che AQM co the co du lieu vet goi tin (out_ip_header_*.csv) - script
# tu bo qua (in [warn]) neu file cua co che nao chua ton tai, khong can sua
# code khi chi co mot phan trong so nay o thoi diem chay.
AQM_AVAILABLE = ["droptail", "codel", "fqcodel", "red", "ared", "pie", "cake", "dcodel",
                  "dn_dcod_khong_nhan", "dn_dcod"]
AQM_DISPLAY_NAME = {"droptail": "DropTail", "codel": "CoDel", "fqcodel": "FQ-CoDel",
                     "red": "RED", "ared": "ARED", "pie": "PIE", "cake": "CAKE", "dcodel": "dCoDel",
                     "dn_dcod_khong_nhan": "DN-dCoD (không nhãn)", "dn_dcod": "DN-dCoD"}

USECOLS = ["Timestamp", "event_type", "direction", "src_ip", "dst_ip", "src_port", "dst_port",
           "protocol", "identification", "payload_size", "queue_size", "d_queue_ms"]
DTYPES = {"direction": "float32", "src_ip": "category", "dst_ip": "category", "protocol": "category",
          "event_type": "category", "payload_size": "float32", "queue_size": "float32",
          "d_queue_ms": "float32", "src_port": "float32", "dst_port": "float32"}


def trace_path(aqm, seed, run):
    return f"{TRACE_DIR}/out_ip_header_{aqm}_seed{seed}_run{run}.csv"


def _keep_bottleneck_direction(df):
    """Chi giu su kien hang doi cua HUONG 0 (chieu du lieu qua nut that co chai).
    Huong 1 trong vet NS-3 la chieu ACK (hang doi luon rong: tre 0, do dai 0-1,
    ~17% so su kien) - gop vao se LAM LOANG tre/hang doi/mat goi cua cac AQM NS-3 va
    dCoDel, trong khi DN-dCoD chi xuat huong 0 (khong bi loang) -> so sanh lech.
    TX/RX khong co huong nen giu nguyen."""
    q_evt = df["event_type"].isin(["ENQUEUE", "DEQUEUE", "DROP"])
    return df[~q_evt | (df["direction"] == 0)]


def _clean_queue_events(df):
    """Sua loi ghi vet cua NS-3 voi cac qdisc co DROP SAU DEQUEUE (CoDel, FQ-CoDel, CAKE).

    Voi cac co che nay, goi bi tha o DAU hang doi (head drop) duoc ghi HAI LAN: mot dong DEQUEUE
    roi mot dong DROP, cung Timestamp va cung (src_ip, dst_ip, identification). Hau qua neu khong
    xu ly: (a) thong luong va Jain bi tinh ca goi da bi tha (CoDel do duoc > 10 Mbps tren lien ket
    10 Mbps); (b) do tre tinh ca goi bi tha; (c) mau so ty le mat goi bi cong them cac goi da duoc
    dem o ENQUEUE; (d) cot queue_size bi tru hai lan nen thap hon thuc te.
    Ham gan 2 cot:
      _drop_after_deq : DROP co dong DEQUEUE trung khoa (goi da vao hang doi roi moi bi tha)
      _deq_was_dropped: DEQUEUE co dong DROP trung khoa (goi KHONG duoc chuyen di thuc su)
    Chi doi chieu khi identification khong rong (vet offline khong co cot nay thi giu nguyen).
    DropTail/PIE/RED tha TRUOC khi vao hang doi nen khong bi anh huong."""
    df = df.copy()
    df["_drop_after_deq"] = False
    df["_deq_was_dropped"] = False
    if "identification" not in df.columns:
        return df
    key = ["Timestamp", "src_ip", "dst_ip", "identification"]
    is_dq = (df["event_type"] == "DEQUEUE") & df["identification"].notna()
    is_dr = (df["event_type"] == "DROP") & df["identification"].notna()
    if not is_dq.any() or not is_dr.any():
        return df
    # VET OFFLINE (dCoDel, DN-dCoD): identification luon = 0 -> khoa (Timestamp, src, dst, 0) se TRUNG
    # NHAM giua cac goi khac nhau cung luong cung thoi diem. Vet nay khong ghi 2 lan nen khong can doi chieu.
    if (df.loc[is_dq | is_dr, "identification"] == 0).all():
        df.attrs["offline"] = True
        return df

    def _h(mask):
        k = df.loc[mask, key].copy()
        k["src_ip"] = k["src_ip"].astype(str)
        k["dst_ip"] = k["dst_ip"].astype(str)
        return pd.util.hash_pandas_object(k, index=False).to_numpy()

    h_dq, h_dr = _h(is_dq), _h(is_dr)
    df.loc[is_dr, "_drop_after_deq"] = np.isin(h_dr, h_dq)
    df.loc[is_dq, "_deq_was_dropped"] = np.isin(h_dq, h_dr)
    return df


def compute_metrics_one_run(csv_path):
    """Doc 1 file vet goi tin, tra ve dict 5 chi tieu cho 1 lan chay (1 AQM, 1 seed, 1 run).

    QUAN TRONG: payload_size KHONG duoc ghi truc tiep tai ENQUEUE/DEQUEUE
    trong file goc (chi TX/RX moi co) - phai ghep noi (merge_asof theo cap
    src_ip/dst_ip, gan nhat ve thoi gian) de lay kich thuoc goi.

    Thong luong va Jain's Index tinh tu su kien DEQUEUE (goi da qua duoc
    hang doi that co chai), KHONG dung RX - vi Config::Connect trong file
    .cc bat Ipv4L3Protocol/Rx tren "/NodeList/*/" (MOI nut, ke ca cac nut
    trung gian chuyen tiep goi), khien tong byte RX toan mang VUOT QUA
    nang luc lien ket that co chai that (vi du do duoc ~32Mbps trong khi
    lien ket chi 10Mbps) - khong phai loi du lieu, ma la pham vi do khac
    voi dinh nghia "thong luong qua lien ket that co chai" cua muc 4.3.2."""
    usecols_extra = USECOLS + ["src_port", "dst_port"]
    df = pd.read_csv(csv_path, usecols=usecols_extra, dtype=DTYPES)
    df = _keep_bottleneck_direction(df)
    df = _clean_queue_events(df)

    dq_all = df[df["event_type"] == "DEQUEUE"]
    drops = df[df["event_type"] == "DROP"]
    # Goi THUC SU duoc chuyen di = DEQUEUE khong bi tha ngay sau do
    dequeue = dq_all[~dq_all["_deq_was_dropped"]].copy()

    # --- Do tre hang doi trung binh: d_queue_ms (sojourn) cua goi duoc chuyen di ---
    avg_delay_ms = float(dequeue["d_queue_ms"].mean()) if len(dequeue) else float("nan")

    # --- Do dai hang doi trung binh theo dinh luat Little: L = (tong thoi gian cho cua moi goi
    #     da roi hang doi) / thoi luong. Khong dung cot queue_size vi o vet NS-3 cot nay bi
    #     tru qua nhieu khi co DROP (goi tha truoc khi vao hang doi, hoac DROP sau DEQUEUE). ---
    if len(dq_all):
        _dur = float(dq_all["Timestamp"].max() - dq_all["Timestamp"].min())
        _soj = float(dq_all["d_queue_ms"].clip(lower=0).sum())
        # goi bi tha trong hang doi (vet offline): DROP khong co DEQUEUE trung khoa, co the mang d_queue_ms
        _soj += float(drops.loc[~drops["_drop_after_deq"], "d_queue_ms"].fillna(0).clip(lower=0).sum())
        avg_qlen = (_soj / 1000.0) / _dur if _dur > 0 else float("nan")
    else:
        avg_qlen = float("nan")

    # --- Ty le mat goi: tong DROP / so goi den hang doi.
    #     So goi den = ENQUEUE + DROP xay ra TRUOC khi vao hang doi (DROP sau DEQUEUE da nam trong ENQUEUE) ---
    n_enqueue = int((df["event_type"] == "ENQUEUE").sum())
    n_drop = int(len(drops))
    n_drop_before = int((~drops["_drop_after_deq"]).sum())
    if df.attrs.get("offline", False):
        # vet offline: dCoDel ghi goi bi tha o DAU hang doi la DROP nhung goi do da co ENQUEUE (khong co
        # DEQUEUE kem theo) -> ENQUEUE + DROP dem trung -> ty le mat goi bi THAP gia tao. Moi goi den hoac
        # roi hang doi (DEQUEUE) hoac bi tha (DROP) nen so goi den = DEQUEUE + DROP.
        loss_rate = n_drop / max(1, len(dq_all) + n_drop)
    else:
        loss_rate = n_drop / max(1, (n_enqueue + n_drop_before))

    # --- Ghep noi payload_size cho DEQUEUE tu TX gan nhat cung cap src_ip/dst_ip ---
    tx = df[df["event_type"] == "TX"][["Timestamp", "src_ip", "dst_ip", "payload_size"]].copy()
    tx = tx.rename(columns={"payload_size": "payload_size_tx"}).sort_values("Timestamp")
    dequeue = dequeue.sort_values("Timestamp")
    dequeue = pd.merge_asof(dequeue, tx, on="Timestamp", by=["src_ip", "dst_ip"],
                             direction="nearest", tolerance=0.02)
    # Neu DEQUEUE da co payload_size rieng (vet mo phong offline: dCoDel, DN-dCoD,
    # UKM) thi DUNG TRUC TIEP - ghep TX chi la phuong an cho vet NS-3 (khong co cot nay).
    _own = pd.to_numeric(dequeue["payload_size"], errors="coerce")
    dequeue["payload_size_tx"] = _own.where(_own.notna(), dequeue["payload_size_tx"])
    dequeue["payload_size_tx"] = dequeue["payload_size_tx"].fillna(
        dequeue["payload_size_tx"].median() if dequeue["payload_size_tx"].notna().any() else 0)

    # --- Thong luong: tong byte DEQUEUE / thoi luong quan sat duoc (qua lien ket that co chai) ---
    if len(dequeue):
        total_bytes = float(dequeue["payload_size_tx"].sum())
        duration_s = float(dequeue["Timestamp"].max() - dequeue["Timestamp"].min())
        throughput_mbps = (total_bytes * 8 / 1e6) / duration_s if duration_s > 0 else float("nan")
    else:
        throughput_mbps = float("nan")

    # --- Jain's Fairness Index: theo tong byte DEQUEUE cua tung cap (src_ip, dst_ip) ---
    # (khong co src_port/dst_port o ENQUEUE/DEQUEUE trong file goc nen chi
    # dinh danh duoc luong theo cap IP, khong phai 5-tuple day du)
    if len(dequeue):
        flow_bytes = dequeue.groupby(["src_ip", "dst_ip"], observed=True)["payload_size_tx"].sum()
        flow_bytes = flow_bytes[flow_bytes > 0].to_numpy(dtype=np.float64)
        if len(flow_bytes) > 0:
            jain = float((flow_bytes.sum() ** 2) / (len(flow_bytes) * np.sum(flow_bytes ** 2)))
        else:
            jain = float("nan")
        n_flows = int(dequeue.groupby(["src_ip", "dst_ip"], observed=True).ngroups)
    else:
        jain = float("nan")
        n_flows = 0

    return dict(avg_delay_ms=avg_delay_ms, avg_qlen=avg_qlen, loss_rate=loss_rate,
                throughput_mbps=throughput_mbps, jain_index=jain,
                n_packets=len(df), n_flows=n_flows)


def run_comparison(seeds, runs):
    rows = []
    for aqm in AQM_AVAILABLE:
        for seed in seeds:
            for run in runs:
                path = trace_path(aqm, seed, run)
                if not Path(path).exists():
                    print(f"[warn] khong tim thay: {path}")
                    continue
                print(f"[{aqm} seed{seed} run{run}] dang doc {path} ...")
                m = compute_metrics_one_run(path)
                m.update(aqm=aqm, seed=seed, run=run)
                rows.append(m)
                print(f"    delay={m['avg_delay_ms']:.3f}ms qlen={m['avg_qlen']:.2f} "
                      f"loss={m['loss_rate']*100:.3f}% throughput={m['throughput_mbps']:.2f}Mbps "
                      f"jain={m['jain_index']:.4f} (n_flows={m['n_flows']})")
    return pd.DataFrame(rows)


def summarize(df_runs):
    agg = df_runs.groupby("aqm").agg(
        n_runs=("avg_delay_ms", "count"),
        avg_delay_ms_mean=("avg_delay_ms", "mean"), avg_delay_ms_std=("avg_delay_ms", "std"),
        avg_qlen_mean=("avg_qlen", "mean"), avg_qlen_std=("avg_qlen", "std"),
        loss_rate_mean=("loss_rate", "mean"), loss_rate_std=("loss_rate", "std"),
        throughput_mbps_mean=("throughput_mbps", "mean"), throughput_mbps_std=("throughput_mbps", "std"),
        jain_index_mean=("jain_index", "mean"), jain_index_std=("jain_index", "std"),
    ).reset_index()
    order = [a for a in AQM_AVAILABLE if a in agg["aqm"].values]
    agg = agg.set_index("aqm").loc[order].reset_index()
    agg["display_name"] = agg["aqm"].map(AQM_DISPLAY_NAME)
    return agg


def print_table(agg):
    print("\n" + "=" * 110)
    print("BANG SO SANH CAC CO CHE AQM (Thuc nghiem 3)")
    print("=" * 110)
    print(f"{'AQM':<12}{'n':<4}{'Delay(ms)':<18}{'Qlen(goi)':<18}{'Loss(%)':<14}{'Throughput(Mbps)':<20}{'Jain':<10}")
    for _, r in agg.iterrows():
        def fmt(m, s):
            return f"{m:.3f}+/-{s:.3f}" if r["n_runs"] > 1 and not np.isnan(s) else f"{m:.3f}"
        print(f"{r['display_name']:<12}{r['n_runs']:<4}"
              f"{fmt(r['avg_delay_ms_mean'], r['avg_delay_ms_std']):<18}"
              f"{fmt(r['avg_qlen_mean'], r['avg_qlen_std']):<18}"
              f"{fmt(r['loss_rate_mean']*100, r['loss_rate_std']*100):<14}"
              f"{fmt(r['throughput_mbps_mean'], r['throughput_mbps_std']):<20}"
              f"{fmt(r['jain_index_mean'], r['jain_index_std']):<10}")
    print("=" * 110)


def plot_line_per_aqm(agg):
    """Moi co che AQM = 1 bieu do duong (1 hang), truc x la 5 chi tieu, moi
    chi tieu 1 mau rieng. Vi don vi do khac nhau hoan toan (ms, goi, %,
    Mbps, 0-1) nen chuan hoa min-max giua cac AQM ve thang 0-1 truoc khi ve
    chung 1 truc y - ghi ro tren bieu do de khong hieu nham la gia tri
    thuc te."""
    metric_cols = ["avg_delay_ms_mean", "avg_qlen_mean", "loss_rate_mean",
                   "throughput_mbps_mean", "jain_index_mean"]
    metric_labels = ["Độ trễ TB (ms)", "Độ dài hàng đợi TB (gói)", "Mất gói (%)",
                     "Thông lượng (Mbps)", "Jain"]
    metric_colors = ["#c0392b", "#e67e22", "#8e44ad", "#27ae60", "#2980b9"]

    # chuan hoa min-max TUNG chi tieu qua cac AQM (0 = nho nhat, 1 = lon nhat)
    norm = agg.copy()
    for col in metric_cols:
        vmin, vmax = agg[col].min(), agg[col].max()
        norm[col] = 0.5 if vmax == vmin else (agg[col] - vmin) / (vmax - vmin)

    n_aqm = len(agg)
    fig, axes = plt.subplots(n_aqm, 1, figsize=(9, 3.2 * n_aqm), sharex=True)
    if n_aqm == 1:
        axes = [axes]
    x = np.arange(len(metric_cols))

    for i, (ax, (_, row_raw), (_, row_norm)) in enumerate(zip(axes, agg.iterrows(), norm.iterrows())):
        ax.set_facecolor("#f5f5f3")
        ax.grid(True, color="#b5b5b5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        y = [row_norm[c] for c in metric_cols]
        ax.plot(x, y, color="#555555", linewidth=3.5, zorder=2)  # duong noi day, mau trung tinh
        for xi, (c, col) in enumerate(zip(metric_colors, metric_cols)):
            ax.scatter(xi, row_norm[col], color=c, s=180, zorder=3, edgecolor="black", linewidth=1.2)
            raw_val = row_raw[col] * (100 if col == "loss_rate_mean" else 1)
            ax.annotate(f"{raw_val:.2f}", (xi, row_norm[col]), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=9, fontweight="bold", color=c)
        ax.set_ylim(-0.15, 1.25)
        ax.set_yticks([])
        ax.set_ylabel(row_raw["display_name"], fontsize=12, fontweight="bold", rotation=0,
                       ha="right", va="center", labelpad=15)
        if i == n_aqm - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, fontsize=10)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

    plt.suptitle("Thực nghiệm 3 — Hồ sơ hiệu năng từng cơ chế AQM\n"
                 "(vị trí điểm = giá trị chuẩn hoá 0-1 giữa 5 AQM; số ghi trên điểm = giá trị thực)",
                 fontweight="bold", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.show()


def compute_timeseries_one_run(csv_path, bin_seconds=1.0):
    """Doc 1 file vet goi tin, tra ve DataFrame 5 chi tieu tinh theo TUNG
    CUA SO thoi gian (mac dinh 1 giay/cua so) thay vi 1 con so tong hop
    duy nhat - dung de ve bien thien theo thoi gian. Dung lai dung logic
    ghep noi payload_size (qua TX) va DEQUEUE-based throughput/Jain nhu
    compute_metrics_one_run()."""
    usecols_extra = USECOLS + ["src_port", "dst_port"]
    df = pd.read_csv(csv_path, usecols=usecols_extra, dtype=DTYPES)
    df = _keep_bottleneck_direction(df)
    df = _clean_queue_events(df)

    tx = df[df["event_type"] == "TX"][["Timestamp", "src_ip", "dst_ip", "payload_size"]].copy()
    tx = tx.rename(columns={"payload_size": "payload_size_tx"}).sort_values("Timestamp")

    dq_all_ts = df[df["event_type"] == "DEQUEUE"].copy()
    dequeue = dq_all_ts[~dq_all_ts["_deq_was_dropped"]].copy().sort_values("Timestamp")
    dequeue = pd.merge_asof(dequeue, tx, on="Timestamp", by=["src_ip", "dst_ip"],
                             direction="nearest", tolerance=0.02)
    med = dequeue["payload_size_tx"].median() if dequeue["payload_size_tx"].notna().any() else 0
    _own = pd.to_numeric(dequeue["payload_size"], errors="coerce")
    dequeue["payload_size_tx"] = _own.where(_own.notna(), dequeue["payload_size_tx"])
    med = dequeue["payload_size_tx"].median() if dequeue["payload_size_tx"].notna().any() else 0
    dequeue["payload_size_tx"] = dequeue["payload_size_tx"].fillna(med)

    enqueue = df[df["event_type"] == "ENQUEUE"].copy()
    drop = df[df["event_type"] == "DROP"].copy()
    q_samples = dq_all_ts.copy()   # do dai hang doi theo Little: tong thoi gian cho cua goi roi hang doi trong cua so

    for d in (dequeue, enqueue, drop, q_samples):
        d["time_bin"] = (d["Timestamp"] // bin_seconds) * bin_seconds

    max_t = df["Timestamp"].max()
    bins = np.arange(0, max_t + bin_seconds, bin_seconds)
    out = pd.DataFrame({"time_bin": bins})

    delay_by_bin = dequeue.groupby("time_bin")["d_queue_ms"].mean()
    qlen_by_bin = q_samples.groupby("time_bin")["d_queue_ms"].apply(lambda x: x.clip(lower=0).sum() / 1000.0 / bin_seconds)
    enq_by_bin = enqueue.groupby("time_bin").size()
    drop_by_bin = drop.groupby("time_bin").size()
    bytes_by_bin = dequeue.groupby("time_bin")["payload_size_tx"].sum()

    out["avg_delay_ms"] = out["time_bin"].map(delay_by_bin)
    out["avg_qlen"] = out["time_bin"].map(qlen_by_bin)
    n_enq = out["time_bin"].map(enq_by_bin).fillna(0)
    n_drop = out["time_bin"].map(drop_by_bin).fillna(0)
    drop_before = drop[~drop["_drop_after_deq"]].groupby("time_bin").size()
    n_drop_before = out["time_bin"].map(drop_before).fillna(0)
    out["loss_rate"] = (n_drop / (n_enq + n_drop_before).replace(0, np.nan)) * 100
    out["throughput_mbps"] = (out["time_bin"].map(bytes_by_bin).fillna(0) * 8 / 1e6) / bin_seconds

    jain_by_bin = {}
    for t, g in dequeue.groupby("time_bin"):
        flow_bytes = g.groupby(["src_ip", "dst_ip"], observed=True)["payload_size_tx"].sum()
        flow_bytes = flow_bytes[flow_bytes > 0].to_numpy(dtype=np.float64)
        if len(flow_bytes) > 0:
            jain_by_bin[t] = (flow_bytes.sum() ** 2) / (len(flow_bytes) * np.sum(flow_bytes ** 2))
    out["jain_index"] = out["time_bin"].map(jain_by_bin)

    return out


def plot_timeseries_per_metric(timeseries_by_aqm):
    """DAO CHIEU so voi plot_timeseries_per_aqm: thay vi 1 bieu do/AQM (5
    duong chi tieu), gio la 1 bieu do/CHI TIEU (5 bieu do), moi bieu do co
    toi da 8 duong - 1 duong/co che AQM, 8 mau rieng biet. Vi trong 1 bieu
    do chi co DUNG 1 chi tieu (cung 1 don vi) nen dung GIA TRI THUC (khong
    can chuan hoa 0-1/0-10 nhu ban truoc), truc y hien dung don vi vat ly."""
    metric_specs = [
        ("avg_delay_ms", "Độ trễ hàng đợi trung bình", "ms"),
        ("avg_qlen", "Độ dài hàng đợi trung bình", "gói"),
        ("loss_rate", "Tỷ lệ mất gói", "%"),
        ("throughput_mbps", "Thông lượng", "Mbps"),
        ("jain_index", "Jain's Fairness Index", ""),
    ]
    aqm_names = list(timeseries_by_aqm.keys())
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
    aqm_colors = {name: palette[i % len(palette)] for i, name in enumerate(aqm_names)}

    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4.6 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, (col, title, unit) in zip(axes, metric_specs):
        ax.set_facecolor("#f5f5f3")
        ax.grid(True, color="#d6d6d6", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for aqm_name, ts in timeseries_by_aqm.items():
            # loss_rate tra ve tu compute_timeseries_one_run() DA la % san
            # (khac voi cac ham khac tra ve ty le 0-1) - khong nhan lai 100 lan nua
            y = ts[col]
            display_name = AQM_DISPLAY_NAME.get(aqm_name, aqm_name)
            ax.plot(ts["time_bin"], y, color=aqm_colors[aqm_name], linewidth=1.6,
                     label=display_name, zorder=3)
        ylabel = f"{title} ({unit})" if unit else title
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=min(len(aqm_names), 8))

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.suptitle("Thực nghiệm 3 — So sánh các cơ chế AQM theo từng chỉ tiêu, biến thiên theo thời gian\n"
                 "(mỗi biểu đồ 1 chỉ tiêu, giá trị THỰC theo đúng đơn vị; mỗi đường 1 cơ chế AQM)",
                 fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_timeseries_per_aqm(timeseries_by_aqm):
    """Moi co che AQM = 1 bieu do duong, xep dang LUOI (mac dinh 2 cot, so
    hang tu dong theo so AQM - vi du 8 AQM -> 4 hang x 2 cot). Truc x la
    Thoi gian (s), 5 duong mau the hien 5 chi tieu bien thien theo thoi
    gian. Vi don vi khac nhau hoan toan nen chuan hoa min-max TUNG chi
    tieu THEO CHINH chuoi thoi gian cua no (khong so sanh cheo giua cac
    AQM) - truc y hien so 0.00-1.00 kem nhan "Gia tri chuan hoa (0-1)" de
    nguoi doc biet day la thang da chuan hoa, khong phai gia tri tuyet doi
    (gia tri tuyet doi da ghi tren tung diem/trong bang so o cac ham khac)."""
    metric_cols = ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]
    metric_labels = ["Độ trễ (ms)", "Độ dài hàng đợi (gói)", "Mất gói (%)",
                     "Thông lượng (Mbps)", "Jain"]
    metric_colors = ["#c0392b", "#e67e22", "#8e44ad", "#27ae60", "#2980b9"]

    n_aqm = len(timeseries_by_aqm)
    n_cols = 2 if n_aqm > 1 else 1
    n_rows = int(np.ceil(n_aqm / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 3.4 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (ax, (aqm_name, ts)) in enumerate(zip(axes_flat, timeseries_by_aqm.items())):
        ax.set_facecolor("#f5f5f3")
        ax.grid(True, color="#b5b5b5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        x = ts["time_bin"]
        for col, label, color in zip(metric_cols, metric_labels, metric_colors):
            y = ts[col]
            vmin, vmax = y.min(skipna=True), y.max(skipna=True)
            if pd.isna(vmin) or vmax == vmin:
                y_norm = pd.Series(5.0, index=y.index)
            else:
                y_norm = (y - vmin) / (vmax - vmin) * 10.0   # thang 0-10 thay vi 0-1
            ax.plot(x, y_norm, color=color, linewidth=1.1, label=label, zorder=3)
        ax.set_ylim(-0.5, 10.5)
        ax.set_yticks([0.0, 2.5, 5.0, 7.5, 10.0])
        ax.set_yticklabels(["0.00", "2.50", "5.00", "7.50", "10.00"], fontsize=8)
        ax.set_ylabel("Giá trị chuẩn hoá (0–10)", fontsize=9)
        display_name = AQM_DISPLAY_NAME.get(aqm_name, aqm_name)
        ax.set_title(f"Tiêu chí hiệu năng của cơ chế {display_name}", fontweight="bold", fontsize=12)
        duration_s = float(x.max() - x.min()) if len(x) else 0.0
        ax.set_xlabel(f"Time ({duration_s:.0f}s)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=5)

    for ax in axes_flat[n_aqm:]:
        ax.axis("off")


    plt.suptitle("Thực nghiệm 3 — Biến thiên các chỉ tiêu theo thời gian, từng cơ chế AQM\n"
                 "(mỗi đường chuẩn hoá 0-1 theo min-max của CHÍNH chuỗi thời gian đó)",
                 fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_comparison(agg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("avg_delay_ms_mean", "avg_delay_ms_std", "Độ trễ hàng đợi TB (ms)", axes[0, 0], "#c0392b"),
        ("avg_qlen_mean", "avg_qlen_std", "Độ dài hàng đợi TB (gói)", axes[0, 1], "#e67e22"),
        ("loss_rate_mean", "loss_rate_std", "Tỷ lệ mất gói (%)", axes[1, 0], "#8e44ad"),
        ("throughput_mbps_mean", "throughput_mbps_std", "Thông lượng (Mbps)", axes[1, 1], "#27ae60"),
    ]
    x = np.arange(len(agg))
    for mean_col, std_col, title, ax, color in metrics:
        vals = agg[mean_col] * (100 if mean_col == "loss_rate_mean" else 1)
        errs = agg[std_col].fillna(0) * (100 if mean_col == "loss_rate_mean" else 1)
        ax.set_facecolor("#f5f5f3")
        ax.grid(True, color="#b5b5b5", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.bar(x, vals, yerr=errs, capsize=4, color=color, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["display_name"], fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)

    plt.suptitle("Thực nghiệm 3 — So sánh các cơ chế AQM",
                 fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="10", help="vd '10' hoac '10,11,12'")
    ap.add_argument("--runs", type=str, default="1", help="vd '1' hoac '1,2,3'")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--bin_seconds", type=float, default=1.0,
                     help="do rong cua so thoi gian (s) cho bieu do bien thien theo thoi gian")
    cli_args, _unknown = ap.parse_known_args()

    seeds = [int(s) for s in cli_args.seeds.split(",")]
    runs = [int(r) for r in cli_args.runs.split(",")]

    df_runs = run_comparison(seeds, runs)
    if df_runs.empty:
        print("[error] khong doc duoc file nao - kiem tra lai DATA_DIR / ten file")
    else:
        agg = summarize(df_runs)
        print_table(agg)
        out_csv = cli_args.out_csv or f"{DATA_DIR}/results/aqm_comparison_summary.csv"
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out_csv, index=False)
        print(f"\nsaved -> {out_csv}")
        plot_comparison(agg)
        plot_line_per_aqm(agg)

        # bieu do bien thien theo thoi gian (dung seed/run DAU TIEN co du lieu cho moi AQM)
        print("\n[timeseries] dang tinh chuoi thoi gian cho tung AQM ...")
        timeseries_by_aqm = {}
        for aqm in AQM_AVAILABLE:
            for seed in seeds:
                for run in runs:
                    path = trace_path(aqm, seed, run)
                    if Path(path).exists():
                        print(f"  [{aqm}] tinh chuoi thoi gian tu {path} ...")
                        timeseries_by_aqm[aqm] = compute_timeseries_one_run(path, cli_args.bin_seconds)
                        break
                if aqm in timeseries_by_aqm:
                    break
        if timeseries_by_aqm:
            plot_timeseries_per_metric(timeseries_by_aqm)
            plot_timeseries_per_aqm(timeseries_by_aqm)
