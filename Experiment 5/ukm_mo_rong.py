#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UKM: doi chung "chi bo bao ve" + bo bao ve + tu dieu chinh (Buoc 2) tren du lieu UKM, 1 file ket qua.
Dung lai chinh sach UKM da huan luyen (run_ukm_dcod.RESULTS_DIR) va cac vet dCoDel/DN-dCoD co san (runs 1..10).
Ket qua: ukm_mo_rong.csv (hang = run x co che, 5 chi tieu). Bo qua dong da co (chay lai duoc sau khi Colab ngat).

Chay (Colab):
  import os; D="/content/drive/MyDrive/Colab/#sualuanan/data1"
  os.environ["LUANAN_DATA_DIR"]=D; os.environ["LUANAN_TRACE_DIR"]=D+"/data2"; %cd {D}
  !python3 ukm_mo_rong.py --stage run --methods keep_shield dn_dcod_khong_nhan_shield dn_dcod_shield --runs 1 2 3
  !python3 ukm_mo_rong.py --stage report
Online (Buoc 2): --methods dn_dcod_khong_nhan_online dn_dcod_online
"""
import os, sys, time, random, argparse
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", ".")
for p in (DATA_DIR, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
METRICS = ["avg_delay_ms", "avg_qlen", "loss_rate", "throughput_mbps", "jain_index"]
LOWER = {"avg_delay_ms": True, "avg_qlen": True, "loss_rate": True, "throughput_mbps": False, "jain_index": False}
BASE = ["dcodel", "dn_dcod_khong_nhan", "dn_dcod"]
EXTRA = ["keep_shield", "dn_dcod_khong_nhan_shield", "dn_dcod_shield", "dn_dcod_khong_nhan_online", "dn_dcod_online"]
OUT = "ukm_mo_rong.csv"


class KeepAgent:
    def select_action(self, s, greedy=True):
        return 0


class ShieldAgent:
    """Giong eval_da_tai.ShieldAgent: sojourn uoc tinh (state[0]*100 ms) > nguong -> ep DROP."""
    def __init__(self, base, thr_ms):
        self.base, self.thr, self.n_over, self.n_all = base, float(thr_ms), 0, 0

    def select_action(self, s, greedy=True):
        self.n_all += 1
        if s[0] * 100.0 > self.thr:
            self.n_over += 1
            return 1
        return self.base.select_action(s, greedy=greedy)


def per_class_all(ptl, path, full, pair):
    """Chi tieu theo lop + dong 'All' (khong phu thuoc ban phan_tich_theo_lop.py cu hay moi)."""
    t = pd.read_csv(path, usecols=["Timestamp", "event_type", "direction", "src_ip", "dst_ip",
                                   "src_port", "dst_port", "payload_size", "d_queue_ms"])
    t = t[(t["direction"] == 0) & t["event_type"].isin(["DEQUEUE", "DROP"])]
    t, cov5, _ = ptl.attach_label(t, full, pair, False)
    dq, dr = t[t["event_type"] == "DEQUEUE"], t[t["event_type"] == "DROP"]
    dur = max(dq["Timestamp"].max() - dq["Timestamp"].min(), 1e-9)
    rows = []
    for c in ["All", "Mice", "Sheep", "Elephant", "Unknown"]:
        q, d = (dq, dr) if c == "All" else (dq[dq["cls"] == c], dr[dr["cls"] == c])
        n = len(q) + len(d)
        rows.append(dict(cls=c, n_pkts=n, n_drop=len(d),
                         delay_ms=q["d_queue_ms"].mean() if len(q) else np.nan,
                         p95_delay_ms=q["d_queue_ms"].quantile(.95) if len(q) else np.nan,
                         loss=len(d) / n if n else np.nan,
                         thr_mbps=q["payload_size"].sum() * 8 / 1e6 / dur))
    return pd.DataFrame(rows), cov5


def run(a):
    import run_ukm_dcod as rk
    import analyze_aqm_comparison as aac
    import dcod_env
    import train_dqn_dcod as tdd
    import run_dn_dcod_labeled as rl
    aac.TRACE_DIR = rk.DATA_DIR
    if a.cpu:
        import torch; tdd.DEVICE = "cpu"; torch.set_num_threads(1)
    if a.policy_dir:
        rk.RESULTS_DIR = a.policy_dir
    shield_ms = a.shield_ms if a.shield_ms is not None else float(dcod_env.T_MAX_TARGET_MS)
    global OUT
    lop = (a.stage == "lop")
    if lop:
        import phan_tich_theo_lop as ptl
        OUT = a.out or "ukm_lop.csv"
    res = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    done = set(zip(res["seed"], res["aqm"])) if len(res) else set()
    ag_n = rk._load_policy("dn_dcod_khong_nhan", "none")
    ag_l = rk._load_policy("dn_dcod", "external")
    for r in a.runs:
        rk.UKM_RUN = r
        if not os.path.exists(rk.trace_csv()) or not os.path.exists(rk.labels_csv()):
            print(f"[run {r}] thieu vet/nhan UKM -> bo qua"); continue
        for aqm in BASE + a.methods:
            if (r, aqm) in done:
                continue
            t0 = time.time()
            if aqm in BASE:
                p = aac.trace_path(aqm, rk.UKM_SEED, r)
                if not os.path.exists(p):
                    print(f"[thieu] {p}"); continue
                tmp = None
            else:
                lab = aqm in ("dn_dcod_shield", "dn_dcod_online")
                fn = rl.build_label_provider(rk.labels_csv()) if lab else None
                env = rk.make_env("external" if lab else "none", a.rate_mbps, fn)
                if aqm == "keep_shield":
                    ag = ShieldAgent(KeepAgent(), shield_ms)
                elif aqm.endswith("_shield"):
                    ag = ShieldAgent(ag_l if lab else ag_n, shield_ms)
                else:
                    import torch
                    from online_adapt import OnlineAgent
                    random.seed(r); np.random.seed(r); torch.manual_seed(r)
                    ag = OnlineAgent(rk._load_policy("dn_dcod" if lab else "dn_dcod_khong_nhan", "external" if lab else "none"),
                                     eps=a.online_eps, lr=a.online_lr, update_every=a.online_every,
                                     window=a.online_window, margin=a.online_margin, rollback=not a.no_rollback)
                    ag.attach(env)
                p = os.path.join(a.tmp_dir, f"ukm_{aqm}_r{r}.csv"); os.makedirs(a.tmp_dir, exist_ok=True)
                env.run_greedy_and_export(ag, p)
                if hasattr(ag, "summary"):
                    print("    " + ag.summary())
                if isinstance(ag, ShieldAgent):
                    print(f"    [shield] ep DROP {ag.n_over}/{ag.n_all} goi ({100*ag.n_over/max(ag.n_all,1):.2f}%)")
                tmp = p
            if lop:
                full, pair = ptl.load_labels(rk.labels_csv())
                df, cov5 = per_class_all(ptl, p, full, pair)
                df["seed"], df["aqm"] = r, aqm
                res = pd.concat([res, df], ignore_index=True)
                res.to_csv(OUT, index=False); done.add((r, aqm))
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)
                sel = df[df["cls"] == "All"]
                if sel.empty:
                    print(f"[canh bao] khong co dong 'All'; cac lop: {list(df['cls'])}, cot: {list(df.columns)}")
                    sel = df
                a0 = sel.iloc[0]
                print(f"[run {r:>2} | {aqm:26s}] nhan khop {cov5:5.1%} | delay={a0['delay_ms']:.2f} loss={100*a0['loss']:.2f}% ({time.time()-t0:.0f}s)")
                continue
            m = aac.compute_metrics_one_run(p)
            row = dict(seed=r, aqm=aqm, **{k: m[k] for k in METRICS})
            res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
            res.to_csv(OUT, index=False); done.add((r, aqm))
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            print(f"[run {r:>2} | {aqm:26s}] delay={m['avg_delay_ms']:.2f} ms loss={100*m['loss_rate']:.2f}% "
                  f"thr={m['throughput_mbps']:.3f} ({time.time()-t0:.0f}s)")
    print(f"\nXong -> {OUT}. Buoc tiep: --stage report")


def report_lop(a):
    """Theo lop (Mice/Sheep/Elephant) tren UKM: mean±SD + Wilcoxon chinh xac, Holm tren 3 cap/ho."""
    from scipy.stats import wilcoxon
    res = pd.read_csv("ukm_lop.csv")
    LW = {"delay_ms": True, "p95_delay_ms": True, "loss": True, "thr_mbps": False}
    PRIMARY = {("Mice", "delay_ms"), ("Elephant", "loss"), ("Elephant", "thr_mbps")}
    FAM = [("A. dong bang", [("dn_dcod", "dcodel"), ("dn_dcod_khong_nhan", "dcodel"), ("dn_dcod", "dn_dcod_khong_nhan")]),
           ("B. + bo bao ve", [("dn_dcod_shield", "dcodel"), ("dn_dcod_khong_nhan_shield", "dcodel"), ("dn_dcod_shield", "dn_dcod_khong_nhan_shield")]),
           ("F. doi chung chi bo bao ve", [("dn_dcod_shield", "keep_shield"), ("dn_dcod_khong_nhan_shield", "keep_shield"), ("keep_shield", "dcodel")])]
    share = res[(res.aqm == "dcodel") & (res.cls != "All")].groupby("cls")["n_pkts"].mean()
    print("So goi TB moi lop (dCoDel):", {k: int(v) for k, v in share.items()})
    for name, prs in FAM:
        cols = sorted({k for p in prs for k in p})
        print(f"\n################ Ho {name} ################")
        for cls, mets in [("All", ["delay_ms", "p95_delay_ms", "loss", "thr_mbps"]), ("Mice", ["delay_ms", "loss", "thr_mbps"]),
                          ("Sheep", ["delay_ms", "loss", "thr_mbps"]), ("Elephant", ["delay_ms", "loss", "thr_mbps"])]:
            print(f"\n---- Lop {cls} ----")
            for m in mets:
                w = res[res.cls == cls].pivot(index="seed", columns="aqm", values=m)
                if not all(c in w for c in cols):
                    continue
                w = w.dropna(subset=cols)
                if len(w) < 7:
                    continue
                tag = "(*)" if (cls, m) in PRIMARY else "   "
                print(f"{tag} {m:13s} " + "  ".join(f"{c}={w[c].mean():.4f}±{w[c].std():.4f}" for c in cols))
                ps, info = [], []
                for x, y in prs:
                    d = w[x] - w[y]
                    b = int(((d < 0) if LW[m] else (d > 0)).sum())
                    ps.append(1.0 if (d == 0).all() else wilcoxon(w[x], w[y], method="exact").pvalue)
                    info.append((x, y, d.mean(), b))
                for (x, y, dm, b), q in zip(info, holm(ps)):
                    print(f"        {x} vs {y}: hieu TB={dm:+.4f}, {x} tot hon {b}/{len(w)}, p_Holm={q:.4f}")


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); cur = 0
    for k, i in enumerate(o):
        cur = max(cur, (m - k) * ps[i]); adj[i] = min(1, cur)
    return adj


def report(a):
    from scipy.stats import wilcoxon
    res = pd.read_csv(OUT)
    FAM = [("A. dong bang", [("dn_dcod", "dcodel"), ("dn_dcod_khong_nhan", "dcodel"), ("dn_dcod", "dn_dcod_khong_nhan")]),
           ("B. + bo bao ve", [("dn_dcod_shield", "dcodel"), ("dn_dcod_khong_nhan_shield", "dcodel"), ("dn_dcod_shield", "dn_dcod_khong_nhan_shield")]),
           ("F. doi chung chi bo bao ve", [("dn_dcod_shield", "keep_shield"), ("dn_dcod_khong_nhan_shield", "keep_shield"), ("keep_shield", "dcodel")]),
           ("D. tu dieu chinh", [("dn_dcod_online", "dcodel"), ("dn_dcod_khong_nhan_online", "dcodel"), ("dn_dcod_online", "dn_dcod_khong_nhan_online")]),
           ("E. tu dieu chinh so voi dong bang", [("dn_dcod_online", "dn_dcod"), ("dn_dcod_khong_nhan_online", "dn_dcod_khong_nhan"), ("dn_dcod_online", "dn_dcod_khong_nhan_online")])]
    for name, prs in FAM:
        cols = sorted({k for p in prs for k in p})
        print(f"\n################ Ho {name} ################")
        for m in METRICS:
            w = res.pivot(index="seed", columns="aqm", values=m)
            if not all(c in w for c in cols):
                continue
            w = w.dropna(subset=cols)
            if len(w) < 3:
                continue
            print(f"\n{m} (n={len(w)}, {'nho tot' if LOWER[m] else 'lon tot'})  " +
                  "  ".join(f"{c}={w[c].mean():.4f}±{w[c].std():.4f}" for c in cols))
            ps, info = [], []
            for x, y in prs:
                d = w[x] - w[y]
                b = int(((d < 0) if LOWER[m] else (d > 0)).sum())
                ps.append(1.0 if (d == 0).all() else wilcoxon(w[x], w[y], method="exact").pvalue)
                info.append((x, y, d.mean(), b))
            for (x, y, dm, b), q in zip(info, holm(ps)):
                print(f"    {x} vs {y}: hieu TB={dm:+.4f}, {x} tot hon {b}/{len(w)}, p_Holm={q:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=["run", "report", "lop", "report_lop"])
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--methods", nargs="+", default=["keep_shield", "dn_dcod_khong_nhan_shield", "dn_dcod_shield"], choices=EXTRA)
    ap.add_argument("--rate_mbps", type=float, default=2.0)
    ap.add_argument("--policy_dir", default=None)
    ap.add_argument("--tmp_dir", default="/tmp/ukm_mo_rong")
    ap.add_argument("--out", default=None, help="file ket qua (mac dinh ukm_mo_rong.csv; moi bo tham so online nen dung file rieng)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shield_ms", type=float, default=None)
    ap.add_argument("--online_eps", type=float, default=0.01)
    ap.add_argument("--online_lr", type=float, default=1e-4)
    ap.add_argument("--online_every", type=int, default=4)
    ap.add_argument("--online_window", type=int, default=5000)
    ap.add_argument("--online_margin", type=float, default=0.25)
    ap.add_argument("--no_rollback", action="store_true")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.stage in ("run", "lop"):
        run(a)
    elif a.stage == "report":
        report(a)
    else:
        report_lop(a)
