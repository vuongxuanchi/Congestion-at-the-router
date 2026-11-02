"""
Thuc nghiem 4 - Danh gia do tin cay thong ke (muc 4.2.4).

Ba che do:
  1) --cls              : so sanh CNN (ReLU) vs WCNN (Wav-KAN) tren 10 seed, du lieu
                          NS-3 va UKM (gia tri tung seed lay tu plot_perseed_accuracy.py).
  2) --wide_csv FILE    : bang "rong" (hang = seed/lan chay, cot = phuong phap/co che).
                          Chay Friedman + hau kiem Nemenyi (alpha=0.05). Dung cho
                          ablation (v1..v4) va cho AQM (moi chi tieu 1 bang).
  3) --collect_ablation : doc 40 log ablation (4 phien ban x 10 seed) -> ablation_wide_accuracy.csv
  4) --collect_aqm      : doc vet out_ip_header_{aqm}_seed{s}_run{r}.csv cho nhieu seed,
                          tinh 5 chi tieu (analyze_aqm_comparison.compute_metrics_one_run),
                          ghi aqm_runs_long.csv va 5 bang rong aqm_wide_<metric>.csv.

Chay tren Colab:
  !python3 stat_tests_exp4.py --cls
  !python3 stat_tests_exp4.py --collect_aqm --seeds 1 2 3 4 5 6 7 8 9 10 --run 1 \
        --aqms droptail codel fqcodel pie cake dcodel dn_dcod_khong_nhan dn_dcod
  !python3 stat_tests_exp4.py --wide_csv aqm_wide_avg_delay_ms.csv --lower_better

Ghi chu phuong phap:
  * Cung chia du lieu (seed chia = 42), 10 seed chi doi khoi tao + thu tu lo => hai mo hinh
    o cung seed KHONG chia se thong tin ghep cap thuc su. Vi vay kiem dinh CHINH la Welch t
    va Mann-Whitney U (doc lap); Wilcoxon/paired-t theo seed chi bao cao tham khao.
  * Friedman/Nemenyi chi hop le khi moi "block" (seed/lan chay) la CUNG MOT kich ban cho moi
    phuong phap (vi du AQM: cung seed sinh luong).
"""
import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

# Gia tri accuracy test tung seed (seed 0..9) - nguon: plot_perseed_accuracy.py
PERSEED = {
    "NS-3": {
        "CNN (ReLU)":     [0.8868, 0.8854, 0.8817, 0.8853, 0.8872, 0.8845, 0.8850, 0.8825, 0.8868, 0.8849],
        "WCNN (Wav-KAN)": [0.8920, 0.8919, 0.8915, 0.8920, 0.8898, 0.8930, 0.8904, 0.8917, 0.8915, 0.8929],
    },
    "UKMDDoSDN": {
        "CNN (ReLU)":     [0.9592, 0.9750, 0.9950, 0.8834, 0.9808, 0.9567, 0.9800, 0.9883, 0.9717, 0.9750],
        "WCNN (Wav-KAN)": [0.9684, 0.9775, 0.9875, 0.9742, 0.9442, 0.9742, 0.9917, 0.9833, 0.9226, 0.9875],
    },
}


# --------------------------------------------------------------------------- 2 nhom
def ci95(x):
    x = np.asarray(x, float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, s, m - h, m + h


def boot_ci_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = rng.choice(b, (n_boot, len(b))).mean(1) - rng.choice(a, (n_boot, len(a))).mean(1)
    return np.percentile(d, [2.5, 97.5])


def two_group_report(a, b, name_a, name_b):
    """b - a (b = mo hinh de xuat). Tra ve dict cac thong ke."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, sa, la, ha = ci95(a)
    mb, sb, lb, hb = ci95(b)
    welch = stats.ttest_ind(b, a, equal_var=False)
    mw = stats.mannwhitneyu(b, a, alternative="two-sided", method="exact")
    pt = stats.ttest_rel(b, a)
    try:
        wx = stats.wilcoxon(b, a, method="exact")
        wx_p = wx.pvalue
    except ValueError:
        wx_p = float("nan")
    sp = np.sqrt((sa ** 2 + sb ** 2) / 2)
    d = (mb - ma) / sp if sp > 0 else float("nan")
    lo, hi = boot_ci_diff(a, b)
    return dict(
        a=name_a, b=name_b, n=len(a),
        mean_a=ma, sd_a=sa, ci_a=(la, ha), mean_b=mb, sd_b=sb, ci_b=(lb, hb),
        diff=mb - ma, diff_boot_ci=(lo, hi),
        welch_t=welch.statistic, welch_p=welch.pvalue,
        mw_U=mw.statistic, mw_p=mw.pvalue,
        paired_t=pt.statistic, paired_p=pt.pvalue, wilcoxon_p=wx_p,
        cohen_d=d, wins=int((b > a).sum()),
        levene_p=stats.levene(a, b).pvalue,
        shapiro_p_a=stats.shapiro(a).pvalue, shapiro_p_b=stats.shapiro(b).pvalue,
    )


def print_two_group(ds, r):
    print(f"\n=== {ds}: {r['b']} so voi {r['a']} (n = {r['n']} seed) ===")
    print(f"  {r['a']:<16s} {r['mean_a']:.4f} ± {r['sd_a']:.4f}   CI95% [{r['ci_a'][0]:.4f}; {r['ci_a'][1]:.4f}]")
    print(f"  {r['b']:<16s} {r['mean_b']:.4f} ± {r['sd_b']:.4f}   CI95% [{r['ci_b'][0]:.4f}; {r['ci_b'][1]:.4f}]")
    print(f"  Chenh lech trung binh = {r['diff']*100:+.2f} diem %, bootstrap CI95% "
          f"[{r['diff_boot_ci'][0]*100:+.2f}; {r['diff_boot_ci'][1]*100:+.2f}] diem %")
    print(f"  Welch t = {r['welch_t']:.3f}, p = {r['welch_p']:.4f}  |  Mann-Whitney U = {r['mw_U']:.1f}, p = {r['mw_p']:.4f}")
    print(f"  (tham khao, ghep cap theo seed) paired-t p = {r['paired_p']:.4f}, Wilcoxon p = {r['wilcoxon_p']:.4f}, "
          f"WCNN thang {r['wins']}/{r['n']} seed")
    print(f"  Cohen d = {r['cohen_d']:.2f}  |  Levene (phuong sai bang nhau) p = {r['levene_p']:.3f}  |  "
          f"Shapiro p: {r['shapiro_p_a']:.3f} / {r['shapiro_p_b']:.3f}")


def run_cls():
    rows = []
    for ds, d in PERSEED.items():
        r = two_group_report(d["CNN (ReLU)"], d["WCNN (Wav-KAN)"], "CNN (ReLU)", "WCNN (Wav-KAN)")
        print_two_group(ds, r)
        rows.append(dict(dataset=ds, **{k: v for k, v in r.items() if not isinstance(v, tuple)},
                         ci_a_lo=r["ci_a"][0], ci_a_hi=r["ci_a"][1], ci_b_lo=r["ci_b"][0], ci_b_hi=r["ci_b"][1],
                         diff_ci_lo=r["diff_boot_ci"][0], diff_ci_hi=r["diff_boot_ci"][1]))
    out = "exp4_cnn_vs_wcnn_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


# --------------------------------------------------------------------------- Friedman / Nemenyi
def friedman_nemenyi(wide, higher_better=True):
    """wide: DataFrame hang=block (seed), cot=phuong phap. Tra ve (friedman, avg_rank, nemenyi_p)."""
    wide = wide.dropna()
    n, k = wide.shape
    arr = wide.to_numpy(float)
    fr = stats.friedmanchisquare(*[arr[:, j] for j in range(k)])
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r if higher_better else r), 1, arr)
    avg = pd.Series(ranks.mean(0), index=wide.columns)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    p = pd.DataFrame(np.nan, index=wide.columns, columns=wide.columns)
    for i, j in itertools.combinations(range(k), 2):
        q = abs(avg.iloc[i] - avg.iloc[j]) / se * np.sqrt(2)
        p.iloc[i, j] = p.iloc[j, i] = float(stats.studentized_range.sf(q, k, np.inf))
    cd = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) * se
    kendall_w = fr.statistic / (n * (k - 1))
    return fr, avg, p, cd, kendall_w, n, k


# Cac cap so sanh DINH TRUOC, chia theo HO (Holm chi ap dung trong tung ho):
#  - ho offline: dCoDel, DN-dCoD (khong nhan), DN-dCoD cung chay offline tren cung vet -> so sanh cong bang
#  - ho NS-3 truc tiep: CoDel va DropTail cung chay NS-3 co phan hoi TCP
CONTRAST_FAMILIES = [
    [("dn_dcod", "dcodel"), ("dn_dcod", "dn_dcod_khong_nhan"), ("dn_dcod_khong_nhan", "dcodel")],
    [("codel", "droptail")],
]


def planned_contrasts(wide, higher_better):
    """Wilcoxon signed-rank (chinh xac, hai phia) cho cac cap dinh truoc + hieu chinh Holm trong tung ho."""
    w = wide.dropna()
    rows = []
    for fam_id, fam in enumerate(CONTRAST_FAMILIES, 1):
        fam_rows = []
        for a, b in fam:
            if a not in w.columns or b not in w.columns:
                continue
            d = (w[a] - w[b]).to_numpy(float)
            wins = int(((d > 0) if higher_better else (d < 0)).sum())
            try:
                pv = float(stats.wilcoxon(d, method="exact").pvalue)
            except Exception:
                pv = float("nan")
            fam_rows.append(dict(ho=fam_id, A=a, B=b, mean_diff=d.mean(), A_tot_hon=f"{wins}/{len(d)}", p=pv))
        ok = sorted([r for r in fam_rows if r["p"] == r["p"]], key=lambda r: r["p"])
        m, run = len(ok), 0.0
        for i, r in enumerate(ok):
            run = max(run, min(1.0, (m - i) * r["p"]))
            r["p_Holm"] = run
        rows += fam_rows
    return rows


def run_wide(path, higher_better):
    wide = pd.read_csv(path, index_col=0)
    dropped = sorted(set(wide.index) - set(wide.dropna().index))
    if dropped:
        print(f"[CANH BAO] bo seed {dropped} vi thieu du lieu cua it nhat mot co che -> n = {len(wide.dropna())}")
    fr, avg, p, cd, w, n, k = friedman_nemenyi(wide, higher_better)
    print(f"\n=== {os.path.basename(path)}: Friedman, n = {n} block, k = {k} phuong phap ===")
    print(f"  chi2_F = {fr.statistic:.3f}, p = {fr.pvalue:.3g}, Kendall W = {w:.3f}")
    print("  Hang trung binh (1 = tot nhat):")
    for m, r in avg.sort_values().items():
        print(f"    {m:<24s} {r:.2f}   mean ± SD = {wide[m].mean():.4f} ± {wide[m].std(ddof=1):.4f}")
    print(f"  Nguong CD Nemenyi (alpha = {ALPHA}) = {cd:.2f}")
    print("  p Nemenyi (cap):")
    print(p.round(4).to_string())
    rows = planned_contrasts(wide, higher_better)
    if rows:
        print("  So sanh cap dinh truoc (Wilcoxon signed-rank chinh xac, hai phia; Holm):")
        for r in rows:
            print(f"    [ho {r['ho']}] {r['A']} vs {r['B']}: hieu TB = {r['mean_diff']:+.4f}, "
                  f"{r['A']} tot hon o {r['A_tot_hon']} seed, p = {r['p']:.4f}, p_Holm = {r.get('p_Holm', float('nan')):.4f}")
    base = os.path.splitext(os.path.basename(path))[0]
    avg.to_csv(f"exp4_{base}_avgrank.csv", header=["avg_rank"])
    p.to_csv(f"exp4_{base}_nemenyi_p.csv")


# --------------------------------------------------------------------------- thu thap ablation (Thuc nghiem 2)
ABL_VERSIONS = ["v1_base", "v2_lr_sched", "v3_ema", "v4_multiwave"]


def collect_ablation(data_dir, n_seeds=10):
    """Doc ablation_{version}_seed{s}.json (train_wcnn_ablation.py) -> bang rong accuracy/F1 macro."""
    import json
    log_dir = os.path.join(data_dir, "results", "ns3-6", "training_logs")
    acc, f1 = {}, {}
    for v in ABL_VERSIONS:
        for sd in range(n_seeds):
            fp = os.path.join(log_dir, f"ablation_{v}_seed{sd}.json")
            if not os.path.exists(fp):
                print(f"[thieu] {fp}")
                continue
            d = json.load(open(fp))
            acc.setdefault(v, {})[sd] = d["test_acc"]
            f1.setdefault(v, {})[sd] = d["test_metrics"]["f1_macro"]
    pd.DataFrame(acc).rename_axis("seed").to_csv("ablation_wide_accuracy.csv")
    pd.DataFrame(f1).rename_axis("seed").to_csv("ablation_wide_f1_macro.csv")
    print("saved -> ablation_wide_accuracy.csv, ablation_wide_f1_macro.csv")


# --------------------------------------------------------------------------- thu thap AQM nhieu seed
def collect_aqm(seeds, run, aqms, data_dir, tag=""):
    sys.path.insert(0, data_dir)
    import importlib
    importlib.invalidate_caches()
    from analyze_aqm_comparison import compute_metrics_one_run, trace_path
    rows = []
    for s in seeds:
        for a in aqms:
            p = trace_path(a, s, run)
            if not os.path.exists(p):
                print(f"[thieu] {p}")
                continue
            m = compute_metrics_one_run(p)
            rows.append(dict(aqm=a, seed=s, **{k: m[k] for k in
                            ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]}))
            print(f"[ok] {a} seed {s}")
    long = pd.DataFrame(rows)
    long.to_csv(f"aqm_runs_long{tag}.csv", index=False)
    for m in ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]:
        long.pivot(index="seed", columns="aqm", values=m).to_csv(f"aqm_wide_{m}{tag}.csv")
    print(f"saved -> aqm_runs_long{tag}.csv + aqm_wide_<metric>{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", action="store_true")
    ap.add_argument("--wide_csv", default=None)
    ap.add_argument("--lower_better", action="store_true", help="chi tieu cang nho cang tot (tre, mat goi, hang doi)")
    ap.add_argument("--collect_aqm", action="store_true")
    ap.add_argument("--collect_ablation", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--tag", default="", help="hau to ten file dau ra, vd _offline")
    ap.add_argument("--aqms", nargs="+",
                    default=["droptail", "codel", "fqcodel", "pie", "cake", "dcodel", "dn_dcod_khong_nhan", "dn_dcod"])
    ap.add_argument("--data_dir", default=os.environ.get("LUANAN_DATA_DIR",
                    "/content/drive/MyDrive/Colab/#sualuanan/data1"))
    a, _ = ap.parse_known_args()
    if a.cls:
        run_cls()
    if a.collect_ablation:
        collect_ablation(a.data_dir)
    if a.collect_aqm:
        collect_aqm(a.seeds, a.run, a.aqms, a.data_dir, a.tag)
    if a.wide_csv:
        run_wide(a.wide_csv, higher_better=not a.lower_better)
    if not (a.cls or a.collect_aqm or a.collect_ablation or a.wide_csv):
        ap.print_help()
