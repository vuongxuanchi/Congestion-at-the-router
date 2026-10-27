"""
Random Forest / XGBoost / SVM baselines for "Thuc nghiem 1" (Bang so sanh
mo hinh phan loai luu luong, muc 4.2.1). Unlike CNN/WCNN, these three use
the WINDOW-LEVEL aggregated feature file (Bo B: ns3_window_*.csv /
ukm_window_2.csv), not the packet-sequence file (Bo A).

Feature set: the thesis text says "Vector 7 dac trung/cua so", but the
actual window-level CSV has 8 common numeric feature columns:
    pkt_count_window, volume_window, packet_size, packet_size_cv,
    ack_ratio, burstiness, peak_rate_100ms, duty_cycle
and correlation with label_code shows NO single column that is clearly
"the" labeling variable on BOTH datasets (NS-3: pkt_count_window/
volume_window dominate; UKM: ack_ratio/burstiness dominate) - so this
script uses all 8 by default. Drop one from FEATURE_COLS below if you
confirm which column the "7 dac trung" wording refers to.

Uses the CSV's own "split" column (train/test) rather than re-deriving a
GroupShuffleSplit, since that split already exists in the file. A small
validation slice is carved out of "train" (by flow_id) only for
XGBoost's early stopping - RF and SVM do not use it.

Usage:
  python3 train_classical_ml.py --dataset ns3 --n_seeds 10
  python3 train_classical_ml.py --dataset ukm --n_seeds 10
"""
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb

# ============================== CONFIG ======================================

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
NS3_WINDOW_FILE = os.environ.get("NS3_WINDOW_FILE", f"{DATA_DIR}/ns3_window_1.csv")
UKM_WINDOW_FILE = os.environ.get("UKM_WINDOW_FILE", f"{DATA_DIR}/ukm_window_2.csv")
RESULTS_DIR_NS3 = f"{DATA_DIR}/results/ns3-6"
RESULTS_DIR_UKM = f"{DATA_DIR}/results_ukm"

SPLIT_SEED = 42        # same seed used everywhere else -> val carve-out is reproducible
N_SEEDS = 10
N_TREES = 300           # RF and XGBoost, per Chuong 4's model table
LABEL_NAMES = ["Mice", "Sheep", "Elephant"]
LABEL_TO_CODE = {n: i for i, n in enumerate(LABEL_NAMES)}

# ten hien thi dep cho bieu do (khong dung de doi ten file/log, chi de ve)
MODEL_DISPLAY_NAME = {"rf": "RF", "xgboost": "XGBoost", "svm": "SVM"}

# the 8 common window-level feature columns (see module docstring)
FEATURE_COLS = [
    "pkt_count_window", "volume_window", "packet_size", "packet_size_cv",
    "ack_ratio", "burstiness", "peak_rate_100ms", "duty_cycle",
]

DATASETS = {
    "ns3": dict(file=NS3_WINDOW_FILE, results_dir=RESULTS_DIR_NS3, name="NS-3"),
    "ukm": dict(file=UKM_WINDOW_FILE, results_dir=RESULTS_DIR_UKM, name="UKM"),
}

# ============================== DATA PREP ===================================


def prepare_window_data(dataset: str):
    cfg = DATASETS[dataset]
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"[data] reading {cfg['file']} ...")
    df = pd.read_csv(cfg["file"])

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected feature columns in {cfg['file']}: {missing}")
    if "split" not in df.columns:
        raise ValueError(f"{cfg['file']} has no 'split' column - expected train/test already assigned")

    df["label_code"] = df["label"].map(LABEL_TO_CODE)
    if df["label_code"].isna().any():
        bad = df.loc[df["label_code"].isna(), "label"].unique()
        raise ValueError(f"unrecognized label values (not in {LABEL_NAMES}): {bad}")

    # NaN handling: ack_ratio is undefined for UDP flows (no ACKs to count) and
    # packet_size_cv/burstiness/peak_rate_100ms/duty_cycle are undefined for
    # single-packet windows (no variance/rate to compute) - these are
    # legitimate "not applicable" cases, not missing data, so filling with 0
    # is the natural default (0 acks observed / 0 variability possible).
    # RF and XGBoost tolerate NaN natively but SVM does not; filling for ALL
    # THREE keeps the comparison on identical input data.
    na_before = df[FEATURE_COLS].isna().sum()
    if na_before.sum() > 0:
        print(f"[data] filling NaN with 0 (UDP/single-packet windows): "
              f"{dict(na_before[na_before > 0])}")
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    train_full = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # carve a validation slice out of train (by flow_id, so no flow leaks across
    # train/val) - only used for XGBoost early stopping.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=SPLIT_SEED)
    tr_rel, val_rel = next(gss.split(train_full, groups=train_full["flow_id"]))
    train_df = train_full.iloc[tr_rel].reset_index(drop=True)
    val_df = train_full.iloc[val_rel].reset_index(drop=True)

    assert set(train_df["flow_id"]).isdisjoint(set(val_df["flow_id"]))
    assert set(train_full["flow_id"]).isdisjoint(set(test_df["flow_id"]))

    print(f"[data] {cfg['name']}: n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)} "
          f"(features: {FEATURE_COLS})")

    Xtr, ytr = train_df[FEATURE_COLS].to_numpy(dtype=np.float64), train_df["label_code"].to_numpy()
    Xval, yval = val_df[FEATURE_COLS].to_numpy(dtype=np.float64), val_df["label_code"].to_numpy()
    Xte, yte = test_df[FEATURE_COLS].to_numpy(dtype=np.float64), test_df["label_code"].to_numpy()
    return (Xtr, ytr), (Xval, yval), (Xte, yte)


# ============================== METRICS =====================================


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


def format_metrics_line(m):
    return (f"acc={m['accuracy']:.4f} | "
            f"P(macro)={m['precision_macro']:.4f} R(macro)={m['recall_macro']:.4f} F1(macro)={m['f1_macro']:.4f} | "
            f"F1(weighted)={m['f1_weighted']:.4f}")


# ============================== MODELS =======================================


def train_rf(seed, Xtr, ytr):
    model = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model


def train_xgboost(seed, Xtr, ytr, Xval, yval):
    model = xgb.XGBClassifier(
        n_estimators=N_TREES, eval_metric="mlogloss", random_state=seed,
        num_class=len(LABEL_NAMES), objective="multi:softprob",
        early_stopping_rounds=20, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def train_svm(seed, Xtr, ytr, scaler):
    # RBF-SVM has no meaningful seed-dependence for a plain .fit() (no
    # bootstrap/subsampling here) - passing random_state anyway for API
    # consistency and in case probability=True is enabled later.
    model = SVC(kernel="rbf", random_state=seed)
    model.fit(scaler.transform(Xtr), ytr)
    return model


# ============================== MAIN RUN ====================================


def run_one_seed(model_type, seed, data, results_dir, job_prefix=""):
    (Xtr, ytr), (Xval, yval), (Xte, yte) = data
    job_name = f"{job_prefix}{model_type}_seed{seed}"
    t0 = time.time()

    if model_type == "rf":
        model = train_rf(seed, Xtr, ytr)
        y_pred = model.predict(Xte)
    elif model_type == "xgboost":
        model = train_xgboost(seed, Xtr, ytr, Xval, yval)
        y_pred = model.predict(Xte)
    elif model_type == "svm":
        scaler = StandardScaler().fit(Xtr)
        model = train_svm(seed, Xtr, ytr, scaler)
        y_pred = model.predict(scaler.transform(Xte))
    else:
        raise ValueError(model_type)

    metrics = compute_metrics(yte, y_pred)
    elapsed = time.time() - t0
    print(f"[{job_name}] {format_metrics_line(metrics)} ({elapsed:.1f}s)")

    log_path = Path(results_dir) / "training_logs" / f"{job_name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"job_name": job_name, "model_type": model_type, "run_seed": seed,
                                     "test_metrics": metrics}, indent=2))
    return metrics


def summarize(model_types, n_seeds, results_dir, dataset_name, job_prefix=""):
    rows, per_class_rows = [], []
    for model_type in model_types:
        accs, pm, rm, fm, fw = [], [], [], [], []
        per_class_f1 = {c: [] for c in LABEL_NAMES}
        per_class_p = {c: [] for c in LABEL_NAMES}
        per_class_r = {c: [] for c in LABEL_NAMES}
        for s in range(n_seeds):
            log_path = Path(results_dir) / "training_logs" / f"{job_prefix}{model_type}_seed{s}.json"
            if not log_path.exists():
                continue
            m = json.loads(log_path.read_text())["test_metrics"]
            accs.append(m["accuracy"]); pm.append(m["precision_macro"]); rm.append(m["recall_macro"])
            fm.append(m["f1_macro"]); fw.append(m["f1_weighted"])
            for c in LABEL_NAMES:
                per_class_f1[c].append(m["per_class"][c]["f1"])
                per_class_p[c].append(m["per_class"][c]["precision"])
                per_class_r[c].append(m["per_class"][c]["recall"])
        if not accs:
            continue
        rows.append({"model": model_type, "n_seeds": len(accs),
                      "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
                      "precision_macro_mean": np.mean(pm), "precision_macro_std": np.std(pm),
                      "recall_macro_mean": np.mean(rm), "recall_macro_std": np.std(rm),
                      "f1_macro_mean": np.mean(fm), "f1_macro_std": np.std(fm),
                      "f1_weighted_mean": np.mean(fw), "f1_weighted_std": np.std(fw)})
        for c in LABEL_NAMES:
            per_class_rows.append({"model": model_type, "class": c,
                                    "precision_mean": np.mean(per_class_p[c]), "precision_std": np.std(per_class_p[c]),
                                    "recall_mean": np.mean(per_class_r[c]), "recall_std": np.std(per_class_r[c]),
                                    "f1_mean": np.mean(per_class_f1[c]), "f1_std": np.std(per_class_f1[c])})

    if not rows:
        print("[summary] nothing to summarize")
        return None, None

    df = pd.DataFrame(rows)
    df_pc = pd.DataFrame(per_class_rows)

    print("\n" + "=" * 100)
    print(f"SUMMARY (mean +/- std over seeds) -- {dataset_name} -- classical ML baselines")
    print("=" * 100)
    print(f"{'Model':<10}{'n':<4}{'Accuracy':<18}{'Prec(macro)':<18}{'Recall(macro)':<18}{'F1(macro)':<18}{'F1(weighted)':<18}")
    for r in rows:
        print(f"{r['model']:<10}{r['n_seeds']:<4}"
              f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}  "
              f"{r['precision_macro_mean']:.4f}+/-{r['precision_macro_std']:.4f}  "
              f"{r['recall_macro_mean']:.4f}+/-{r['recall_macro_std']:.4f}  "
              f"{r['f1_macro_mean']:.4f}+/-{r['f1_macro_std']:.4f}  "
              f"{r['f1_weighted_mean']:.4f}+/-{r['f1_weighted_std']:.4f}")
    print(f"\nPer-class F1 (mean +/- std):")
    print(f"{'Model':<10}{'Class':<10}{'Precision':<18}{'Recall':<18}{'F1':<18}")
    for r in per_class_rows:
        print(f"{r['model']:<10}{r['class']:<10}"
              f"{r['precision_mean']:.4f}+/-{r['precision_std']:.4f}  "
              f"{r['recall_mean']:.4f}+/-{r['recall_std']:.4f}  "
              f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}")
    print("=" * 100)

    csv_path = Path(results_dir) / f"{job_prefix}classical_ml_summary.csv"
    csv_pc_path = Path(results_dir) / f"{job_prefix}classical_ml_summary_per_class.csv"
    df.to_csv(csv_path, index=False)
    df_pc.to_csv(csv_pc_path, index=False)
    print(f"\nsaved -> {csv_path}\nsaved -> {csv_pc_path}")
    return df, df_pc


def plot_comparison(rows, per_class_rows, dataset_name):
    names = [MODEL_DISPLAY_NAME.get(r["model"], r["model"]) for r in rows]
    accs = [r["accuracy_mean"] for r in rows]
    acc_err = [r["accuracy_std"] for r in rows]
    f1s = [r["f1_macro_mean"] for r in rows]
    f1_err = [r["f1_macro_std"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, accs, w, yerr=acc_err, capsize=3, label="Accuracy", color="steelblue")
    ax.bar(x + w/2, f1s, w, yerr=f1_err, capsize=3, label="F1 (macro)", color="indianred")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Giá trị")
    ax.set_title(f"So sánh Accuracy / F1(macro) — {dataset_name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    n_models = len(rows)
    n_classes = len(LABEL_NAMES)
    w2 = 0.8 / n_classes
    colors = ["tab:blue", "tab:orange", "tab:green"]
    # per_class_rows la 1 danh sach phang [{model, class, f1_mean, f1_std, ...}, ...]
    by_model = {}
    for pcr in per_class_rows:
        by_model.setdefault(pcr["model"], {})[pcr["class"]] = pcr
    for ci, c in enumerate(LABEL_NAMES):
        vals = [by_model[r["model"]][c]["f1_mean"] for r in rows]
        errs = [by_model[r["model"]][c]["f1_std"] for r in rows]
        pos = np.arange(n_models) + (ci - (n_classes - 1) / 2) * w2
        ax2.bar(pos, vals, w2, yerr=errs, capsize=2, label=c, color=colors[ci % len(colors)])
    ax2.set_xticks(np.arange(n_models))
    ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("F1-score")
    ax2.set_title(f"So sánh F1 theo từng lớp — {dataset_name}", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ns3", "ukm"], default="ns3",
                     help="mac dinh 'ns3' - doi thanh 'ukm' neu can chay cho UKM")
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS)
    ap.add_argument("--models", type=str, default="rf,xgboost,svm")
    cli_args, _unknown = ap.parse_known_args()

    cfg = DATASETS[cli_args.dataset]
    data = prepare_window_data(cli_args.dataset)
    model_types = cli_args.models.split(",")

    for model_type in model_types:
        n_runs = cli_args.n_seeds
        if model_type == "svm":
            print(f"[note] SVM (RBF, plain .fit()) has no meaningful seed-dependence - "
                  f"running once instead of {cli_args.n_seeds} times.")
            n_runs = 1
        for seed in range(n_runs):
            run_one_seed(model_type, seed, data, cfg["results_dir"])

    df, df_pc = summarize(model_types, cli_args.n_seeds, cfg["results_dir"], cfg["name"])
    if df is not None:
        plot_comparison(df.to_dict("records"), df_pc.to_dict("records"), cfg["name"])
