import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_NAMES = ["Mice", "Sheep", "Elephant"]
CLASSICAL_MODEL_DISPLAY_NAME = {"rf": "RF", "xgboost": "XGBoost", "svm": "SVM"}


def load_seeded_metrics(results_dir, job_prefix, model_type, n_seeds=1):
    """RF/XGBoost/SVM (train_classical_ml.py) or CNN/WCNN (train_*_multiseed.py):
    training_logs/{job_prefix}{model_type}_seed{s}.json, averaged over seeds.
    Voi n_seeds=1 chi doc dung seed0 (1 lan chay), khong dong qua nhieu file
    seed1..9 du chung van con ton tai tren dia."""
    accs, pm, rm, fm, fw = [], [], [], [], []
    per_class_f1 = {c: [] for c in LABEL_NAMES}
    for s in range(n_seeds):
        p = Path(results_dir) / "training_logs" / f"{job_prefix}{model_type}_seed{s}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["test_metrics"]
        accs.append(m["accuracy"]); pm.append(m["precision_macro"]); rm.append(m["recall_macro"])
        fm.append(m["f1_macro"]); fw.append(m["f1_weighted"])
        for c in LABEL_NAMES:
            per_class_f1[c].append(m["per_class"][c]["f1"])
    if not accs:
        return None
    return {
        "model": model_type, "n": len(accs),
        "accuracy": (np.mean(accs), np.std(accs)),
        "precision_macro": (np.mean(pm), np.std(pm)),
        "recall_macro": (np.mean(rm), np.std(rm)),
        "f1_macro": (np.mean(fm), np.std(fm)),
        "f1_weighted": (np.mean(fw), np.std(fw)),
        "per_class_f1": {c: (np.mean(v), np.std(v)) for c, v in per_class_f1.items()},
    }


def load_frcnc_metrics(log_path, label):
    """FRCNC (train_frcnc*.py): {job_name}_log.json, single run (n=1)."""
    d = json.loads(Path(log_path).read_text())
    m = d["test_metrics"]
    zero = 0.0
    return {
        "model": label, "n": 1,
        "accuracy": (m["accuracy"], zero),
        "precision_macro": (m["precision_macro"], zero),
        "recall_macro": (m["recall_macro"], zero),
        "f1_macro": (m["f1_macro"], zero),
        "f1_weighted": (m["f1_weighted"], zero),
        "per_class_f1": {c: (m["per_class"][c]["f1"], zero) for c in LABEL_NAMES},
    }


def check_test_flow_overlap(bo_a_sequence_csv, bo_b_window_csv, split_seed=42, train_size=0.70):
    """Sanity check: does Bo B's own 'split==test' flow set match the flow set
    that a fresh GroupShuffleSplit(seed=split_seed) would put in Bo A's test
    fold? Prints an overlap percentage - run this once before trusting a
    head-to-head RF/XGBoost/SVM vs CNN/WCNN/FRCNC comparison."""
    from sklearn.model_selection import GroupShuffleSplit
    dfb = pd.read_csv(bo_b_window_csv, usecols=["flow_id", "split"])
    test_flows_b = set(dfb.loc[dfb["split"] == "test", "flow_id"].unique())

    dfa = pd.read_csv(bo_a_sequence_csv, usecols=["flow_id"])
    flows_a = dfa["flow_id"].unique()
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=split_seed)
    train_idx, rest_idx = next(gss1.split(flows_a, groups=flows_a))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=split_seed)
    _, test_rel = next(gss2.split(flows_a[rest_idx], groups=flows_a[rest_idx]))
    test_flows_a = set(flows_a[rest_idx][test_rel])

    inter = test_flows_b & test_flows_a
    print(f"[check] Bo B test flows: {len(test_flows_b)} | Bo A test flows: {len(test_flows_a)}")
    print(f"[check] overlap: {len(inter)} flows "
          f"({100*len(inter)/max(1,len(test_flows_b)):.1f}% of Bo B test, "
          f"{100*len(inter)/max(1,len(test_flows_a)):.1f}% of Bo A test)")
    if len(inter) < 0.9 * min(len(test_flows_a), len(test_flows_b)):
        print("[check] WARNING: overlap is well below 90% - these are largely DIFFERENT "
              "test sets. RF/XGBoost/SVM results are not measured on the same held-out "
              "flows as CNN/WCNN/FRCNC; report them side by side but do not claim a "
              "strictly matched comparison in the thesis text.")
    return inter


def print_and_save_table(rows, dataset_name, out_csv=None):
    print("\n" + "=" * 110)
    print(f"BANG SO SANH MO HINH PHAN LOAI LUU LUONG -- {dataset_name} (Thuc nghiem 1)")
    print("=" * 110)
    header = f"{'Mo hinh':<16}{'n':<4}{'Accuracy':<18}{'P(macro)':<18}{'R(macro)':<18}{'F1(macro)':<18}{'F1(weighted)':<18}"
    print(header)
    for r in rows:
        def fmt(k):
            mean, std = r[k]
            return f"{mean:.4f}+/-{std:.4f}" if r["n"] > 1 else f"{mean:.4f}"
        print(f"{r['model']:<16}{r['n']:<4}{fmt('accuracy'):<18}{fmt('precision_macro'):<18}"
              f"{fmt('recall_macro'):<18}{fmt('f1_macro'):<18}{fmt('f1_weighted'):<18}")

    print(f"\nPer-class F1:")
    print(f"{'Mo hinh':<16}" + "".join(f"{c:<20}" for c in LABEL_NAMES))
    for r in rows:
        line = f"{r['model']:<16}"
        for c in LABEL_NAMES:
            mean, std = r["per_class_f1"][c]
            line += f"{(f'{mean:.4f}+/-{std:.4f}' if r['n'] > 1 else f'{mean:.4f}'):<20}"
        print(line)
    print("=" * 110)

    if out_csv:
        flat = []
        for r in rows:
            row = {"model": r["model"], "n": r["n"]}
            for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
                row[f"{k}_mean"], row[f"{k}_std"] = r[k]
            for c in LABEL_NAMES:
                row[f"f1_{c}_mean"], row[f"f1_{c}_std"] = r["per_class_f1"][c]
            flat.append(row)
        pd.DataFrame(flat).to_csv(out_csv, index=False)
        print(f"\nsaved -> {out_csv}")


def plot_comparison(rows, dataset_name):
    names = [r["model"] for r in rows]
    accs = [r["accuracy"][0] for r in rows]
    acc_err = [r["accuracy"][1] for r in rows]
    f1s = [r["f1_macro"][0] for r in rows]
    f1_err = [r["f1_macro"][1] for r in rows]

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
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Giá trị")
    ax.set_title(f"So sánh Accuracy / F1(macro) — {dataset_name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax2 = axes[1]
    n_models = len(rows)
    n_classes = len(LABEL_NAMES)
    w2 = 0.8 / n_classes
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ci, c in enumerate(LABEL_NAMES):
        vals = [r["per_class_f1"][c][0] for r in rows]
        errs = [r["per_class_f1"][c][1] for r in rows]
        pos = np.arange(n_models) + (ci - (n_classes - 1) / 2) * w2
        ax2.bar(pos, vals, w2, yerr=errs, capsize=2, label=c, color=colors[ci % len(colors)])
    ax2.set_xticks(np.arange(n_models))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylim(0.5, 1.05)
    ax2.set_ylabel("F1-score")
    ax2.set_title(f"So sánh F1 theo từng lớp — {dataset_name}", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os

    # Dan san duong dan theo dung cau truc ban dang dung. Sua truc tiep o day
    # neu can, hoac ghi de bang !python summarize_thuc_nghiem_1.py --arg value.
    DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
    #SCALE20_DIR = "/content/drive/MyDrive/Colab/#sualuanan/scale_20"
    #SCALE6_DIR = "/content/drive/MyDrive/Colab/#sualuanan/scale_6"

    DEFAULT_FRCNC_LOGS = [
        f"NS3-50={DATA_DIR}/results/ns3-6/frcnc/frcnc_ns3_agents50_log.json",
        #f"NS3-20={SCALE20_DIR}/results/frcnc/frcnc_ns3_agents20_log.json",
         #f"NS3-6={SCALE6_DIR}/results/frcnc/frcnc_ns3_agents6_log.json",
    ]

    ap = argparse.ArgumentParser()
    ap.add_argument("--classical_ml_dir", default=f"{DATA_DIR}/results/ns3-6",
                     help="RESULTS_DIR used by train_classical_ml.py (contains training_logs/rf_seed*.json etc.)")
    ap.add_argument("--deep_dir", default=f"{DATA_DIR}/results/ns3-6",
                     help="RESULTS_DIR used by train_ns3_multiseed.py / train_ukm_multiseed.py")
    ap.add_argument("--deep_prefix", default="",
                     help="job_name prefix for CNN/WCNN logs, e.g. 'ukm_' for train_ukm_multiseed.py")
    ap.add_argument("--n_seeds", type=int, default=1,
                     help="mac dinh 1 lan chay (doc dung seed0) cho ca 6 mo hinh; "
                          "doi thanh vd 10 neu sau nay muon tinh lai mean+/-std qua nhieu seed")
    ap.add_argument("--frcnc_logs", nargs="*", default=DEFAULT_FRCNC_LOGS,
                     help="LABEL=path pairs, e.g. NS3-50=/.../frcnc_ns3_agents50_log.json")
    ap.add_argument("--dataset_name", default="NS-3")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--check_overlap", nargs=2, metavar=("BO_A_SEQUENCE_CSV", "BO_B_WINDOW_CSV"),
                     default=None, help="optional: verify Bo A / Bo B test-set flow overlap first")
    # parse_known_args (KHONG dung parse_args): khi dan thang code vao 1 o
    # Colab/Jupyter (thay vi chay !python script.py ...), sys.argv chua san
    # cac flag rieng cua kernel (vi du "-f .../kernel-....json"), khien
    # parse_args() bao loi "required: --classical_ml_dir, --deep_dir" ngay
    # ca khi ban khong co y dinh truyen tham so nao ca.
    args, _unknown = ap.parse_known_args()

    if args.check_overlap:
        check_test_flow_overlap(args.check_overlap[0], args.check_overlap[1])
        print()

    rows = []
    for model_type in ["rf", "xgboost", "svm"]:
        r = load_seeded_metrics(args.classical_ml_dir, "", model_type, n_seeds=args.n_seeds)
        if r:
            r["model"] = CLASSICAL_MODEL_DISPLAY_NAME.get(model_type, model_type)
            rows.append(r)
        else:
            print(f"[warn] no logs found for {model_type} in {args.classical_ml_dir}/training_logs")

    for model_type, label in [("cnn", "CNN (ReLU)"), ("wcnn", "WCNN (Wav-KAN)")]:
        r = load_seeded_metrics(args.deep_dir, args.deep_prefix, model_type, n_seeds=args.n_seeds)
        if r:
            r["model"] = label
            rows.append(r)
        else:
            print(f"[warn] no logs found for {args.deep_prefix}{model_type} in {args.deep_dir}/training_logs")

    for kv in args.frcnc_logs:
        label, path = kv.split("=", 1)
        if not Path(path).exists():
            print(f"[warn] FRCNC log not found: {path}")
            continue
        rows.append(load_frcnc_metrics(path, f"FRCNC ({label})"))

    if not rows:
        print("[error] nothing found to summarize - check the paths above")
    else:
        print_and_save_table(rows, args.dataset_name, args.out_csv)
        plot_comparison(rows, args.dataset_name)
