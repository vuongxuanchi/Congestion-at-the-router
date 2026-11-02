#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chi so theo lop luong (Mice / Sheep / Elephant) tu vet NS-3 offline, khong chay lai mo phong.

Colab:
  %cd {D}
  !python3 phan_tich_theo_lop.py --seeds 1            # kiem tra ti le ghep nhan truoc
  !python3 phan_tich_theo_lop.py                       # 10 seed + Wilcoxon/Holm

Dau vao:  data2/out_ip_header_{aqm}_seed{s}_run1.csv  va  frcnc_labels_droptail_seed{s}_run1.csv
Chi so (huong DROP/DEQUEUE = 0, nut co chai):
  delay_ms  = TB d_queue_ms cua DEQUEUE trong lop
  loss      = DROP / (DEQUEUE + DROP) trong lop
  thr_mbps  = tong payload DEQUEUE cua lop *8 / 1e6 / (max-min Timestamp DEQUEUE toan cuc)
Luu y: replay offline la vong ho (nguon khong giam toc khi mat goi) nen KHONG tinh FCT.
Chi so DINH TRUOC (primary): Mice-delay, Elephant-loss, Elephant-thr. Cac o con lai: tham do.
"""
import argparse, os, sys, itertools
import numpy as np
import pandas as pd

AQMS = ["dcodel", "dn_dcod_khong_nhan", "dn_dcod"]
CLASSES = ["Mice", "Sheep", "Elephant"]
PRIMARY = {("Mice", "delay_ms"), ("Elephant", "loss"), ("Elephant", "thr_mbps")}
LOWER = {"delay_ms": True, "p95_delay_ms": True, "loss": True, "thr_mbps": False}
PAIRS = [("dn_dcod", "dcodel"), ("dn_dcod", "dn_dcod_khong_nhan"), ("dn_dcod_khong_nhan", "dcodel")]


def load_labels(path):
    lb = pd.read_csv(path)
    lb["win"] = np.floor(lb["window_start_s"]).astype(int)
    for c in ("src_port", "dst_port"):
        lb[c] = pd.to_numeric(lb[c], errors="coerce")
    full = lb.drop_duplicates(["src_ip", "dst_ip", "src_port", "dst_port", "win"])
    pair = (lb.groupby(["src_ip", "dst_ip", "win"])["label"]
              .agg(lambda s: s.value_counts().index[0]).rename("label_pair").reset_index())
    return full, pair


def attach_label(df, full, pair, use_pair=False):
    df = df.copy()
    df["win"] = np.floor(df["Timestamp"]).astype(int)
    for c in ("src_port", "dst_port"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df.merge(full[["src_ip", "dst_ip", "src_port", "dst_port", "win", "label"]],
                 on=["src_ip", "dst_ip", "src_port", "dst_port", "win"], how="left")
    m = m.merge(pair, on=["src_ip", "dst_ip", "win"], how="left")
    m["cls"] = (m["label"].fillna(m["label_pair"]) if use_pair else m["label"]).fillna("Unknown")
    cov5 = m["label"].notna().mean()
    covp = m["label_pair"].notna().mean() if m["label"].isna().any() else 0.0
    return m, cov5, m["cls"].ne("Unknown").mean()


def per_class(path, full, pair, use_pair=False):
    t = pd.read_csv(path, usecols=["Timestamp", "event_type", "direction", "src_ip", "dst_ip",
                                   "src_port", "dst_port", "payload_size", "d_queue_ms"])
    t = t[(t["direction"] == 0) & t["event_type"].isin(["DEQUEUE", "DROP"])]
    t, cov5, cov = attach_label(t, full, pair, use_pair)
    dq = t[t["event_type"] == "DEQUEUE"]
    dr = t[t["event_type"] == "DROP"]
    dur = dq["Timestamp"].max() - dq["Timestamp"].min()
    rows = []
    for c in ["All"] + CLASSES + ["Unknown"]:
        q, d = (dq, dr) if c == "All" else (dq[dq["cls"] == c], dr[dr["cls"] == c])
        n = len(q) + len(d)
        rows.append(dict(cls=c, n_pkts=n, n_drop=len(d),
                         delay_ms=q["d_queue_ms"].mean() if len(q) else np.nan,
                         p95_delay_ms=q["d_queue_ms"].quantile(.95) if len(q) else np.nan,
                         loss=len(d) / n if n else np.nan,
                         thr_mbps=q["payload_size"].sum() * 8 / 1e6 / dur))
    return pd.DataFrame(rows), cov5, cov


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k]); out[k] = min(1.0, run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("LUANAN_DATA_DIR", "."))
    ap.add_argument("--trace_sub", default="data2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ukm", action="store_true", help="du lieu UKM: don vi = lan chay (--runs), seed co dinh --ukm_seed, vet nam truc tiep trong data_dir")
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--ukm_seed", type=int, default=91)
    ap.add_argument("--pair_fallback", action="store_true", help="du phong theo cap IP (KHONG khuyen dung: gan sai lop)")
    a = ap.parse_args()
    if a.out is None:
        a.out = "class_metrics_ukm.csv" if a.ukm else "class_metrics_offline.csv"

    recs = []
    units = [(r, a.ukm_seed, r) for r in a.runs] if a.ukm else [(s, s, a.run) for s in a.seeds]
    sub = "" if a.ukm else a.trace_sub
    for key, tseed, run in units:
        fn = (f"frcnc_labels_ukm_seed{tseed}_run{run}.csv" if a.ukm
              else f"frcnc_labels_droptail_seed{tseed}_run{run}.csv")
        cands = [os.path.join(a.data_dir, fn), os.path.join(a.data_dir, a.trace_sub, fn)]
        lp = next((c for c in cands if os.path.exists(c)), None)
        if lp is None:
            import glob
            g = glob.glob(os.path.join(a.data_dir, "**", fn), recursive=True)
            lp = g[0] if g else None
        if lp is None:
            print(f"[bo qua] khong tim thay {fn} trong {a.data_dir}"); continue
        full, pair = load_labels(lp)
        for aqm in AQMS:
            tp = os.path.join(a.data_dir, sub, f"out_ip_header_{aqm}_seed{tseed}_run{run}.csv")
            if not os.path.exists(tp):
                print(f"[bo qua] thieu {tp}"); continue
            df, cov5, cov = per_class(tp, full, pair, a.pair_fallback)
            df["seed"], df["aqm"] = key, aqm
            recs.append(df)
            print(f"{'run' if a.ukm else 'seed'} {key:>2} {aqm:20s} ghep nhan 5-tuple={cov5:6.1%}  khong ghep duoc={1-cov:6.1%}")
    if not recs:
        sys.exit("Khong co du lieu.")
    long = pd.concat(recs, ignore_index=True)
    long.to_csv(a.out, index=False)
    print(f"\nDa luu {a.out}")

    # So goi theo lop (de biet lop nao co du mau)
    share = long[(long.aqm == "dcodel") & (long.cls != "All")].groupby("cls")["n_pkts"].mean()
    print("\nSo goi TB moi seed theo lop (dCoDel):"); print(share.round(0).to_string())
    unk = long[long.cls != "All"].groupby("aqm")[["cls", "n_pkts"]].apply(lambda g: g.loc[g.cls == "Unknown", "n_pkts"].sum() / g["n_pkts"].sum())
    print("Ti le goi khong gan duoc nhan:"); print(unk.map("{:.1%}".format).to_string())

    from scipy.stats import wilcoxon
    print("\n=== TB (SD) theo lop, so sanh Wilcoxon chinh xac 2 phia + Holm (3 cap) ===")
    print("(*) = chi so dinh truoc; con lai la tham do\n")
    for c in ["All"] + CLASSES:
        for m in ["delay_ms", "p95_delay_ms", "loss", "thr_mbps"]:
            w = long[long.cls == c].pivot(index="seed", columns="aqm", values=m).dropna()
            if len(w) < 6:
                continue
            tag = "(*)" if (c, m) in PRIMARY else "   "
            ms = "  ".join(f"{k}={w[k].mean():.4f}±{w[k].std():.4f}" for k in AQMS)
            ps, info = [], []
            for x, y in PAIRS:
                d = w[x] - w[y]
                better = (d < 0) if LOWER[m] else (d > 0)
                p = 1.0 if (d == 0).all() else wilcoxon(w[x], w[y], method="exact").pvalue
                ps.append(p); info.append((x, y, d.mean(), int(better.sum()), len(w)))
            ph = holm(ps)
            print(f"{tag} {c:8s} {m:13s} {ms}")
            for (x, y, dm, nb, n), p, q in zip(info, ps, ph):
                print(f"        {x} vs {y}: hieu TB={dm:+.4f}, {x} tot hon {nb}/{n} seed, p={p:.4f}, p_Holm={q:.4f}")
        print()


if __name__ == "__main__":
    main()
