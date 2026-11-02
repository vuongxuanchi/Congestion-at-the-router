#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiem thu DA TAI (chong tac nghen): replay cung chuoi goi NS-3 voi toc do nut co chai GIAM DAN,
khong huan luyen lai. So sanh 4 phuong phap:
    keep_all  = KEEP moi goi (DropTail dem 300 goi, cung bo may env - moc "khong dieu khien")
    dcodel    = dCoDel offline (simulate_dcodel.py)
    dn_dcod_khong_nhan, dn_dcod = chinh sach DQN da huan luyen (greedy, KHONG hoc them)
Chinh sach duoc huan luyen o 10 Mbps => toc do khac 10 la phep thu TONG QUAT HOA / chong tac nghen.

Ket qua luu tai <policy_dir>/da_tai_metrics.csv (dung CUNG --policy_dir cho ca run va report).

Colab:
  D="/content/drive/MyDrive/Colab/#sualuanan/data1"
  import os; os.environ["LUANAN_DATA_DIR"]=D; os.environ["LUANAN_TRACE_DIR"]=D+"/data2"
  %cd {D}
  !python3 eval_da_tai.py --stage run --rates 10 8 6 --policy_dir {D}/results/dn_dcod_v2
  !python3 eval_da_tai.py --stage report --policy_dir {D}/results/dn_dcod_v2

  - Chay lai duoc sau khi Colab ngat: ket qua tung (toc do, seed, phuong phap) luu trong da_tai_metrics.csv.
  - Toc do 10 Mbps DUNG LAI vet da co (dCoDel, 2 DN-dCoD); chi KEEP-all duoc chay moi.
  - Vet trung gian ghi vao thu muc tam cuc bo roi xoa (khong lam day Drive).
Luu y: replay la VONG HO (nguon khong giam toc khi mat goi) => mat goi o tai nang bi danh gia cao
so voi TCP that; dung de SO SANH giua cac phuong phap, khong phai so tuyet doi.
"""
import os, sys, time, argparse, shutil, random
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", ".")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import phan_tich_theo_lop as ptl

CLASSES = ["Mice", "Sheep", "Elephant"]
AQMS_BASE = ["keep_all", "dcodel", "dn_dcod_khong_nhan", "dn_dcod"]
AQMS = AQMS_BASE + ["keep_shield", "dn_dcod_khong_nhan_shield", "dn_dcod_shield",
                     "dn_dcod_khong_nhan_online", "dn_dcod_online"]   # *_shield = + bo bao ve; *_online = tu dieu chinh khi chay (online_adapt.py)
OUT_CSV = "da_tai_metrics.csv"
BASE_RATE = 10.0


def resolve_out(a):
    """Luu canh chinh sach (policy_dir) neu co; --out ghi de. Tu chuyen file cu o thu muc hien tai."""
    global OUT_CSV
    if a.out:
        OUT_CSV = a.out
    elif a.policy_dir:
        OUT_CSV = os.path.join(a.policy_dir, "da_tai_metrics.csv")
    old = "da_tai_metrics.csv"
    if OUT_CSV != old and not os.path.exists(OUT_CSV) and os.path.exists(old):
        os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
        shutil.copy(old, OUT_CSV)
        print(f"[chuyen] {old} -> {OUT_CSV}")
    print(f"[ket qua] {OUT_CSV}")


class KeepAgent:
    """Luon KEEP (hanh dong 0): chi rot khi hang doi day -> DropTail 300 goi."""
    def select_action(self, s, greedy=True):
        return 0


class ShieldAgent:
    """Bo bao ve toi thieu quanh chinh sach da huan luyen: neu sojourn UOC TINH (state[0]*100, ms)
    vuot nguong thi ep DROP; nguong lay tu hang so thiet ke cua env (T_MAX_TARGET_MS), KHONG chinh
    theo du lieu thu. Con lai giu nguyen quyet dinh cua DQN."""
    def __init__(self, base, thr_ms):
        self.base, self.thr, self.n_over, self.n_all = base, float(thr_ms), 0, 0

    def select_action(self, s, greedy=True):
        self.n_all += 1
        if s[0] * 100.0 > self.thr:
            self.n_over += 1
            return 1
        return self.base.select_action(s, greedy=greedy)


def metrics_from_trace(path, full, pair):
    t = pd.read_csv(path, usecols=["Timestamp", "event_type", "direction", "src_ip", "dst_ip",
                                   "src_port", "dst_port", "payload_size", "d_queue_ms"])
    t = t[(t["direction"] == 0) & t["event_type"].isin(["DEQUEUE", "DROP"])]
    t, _, _ = ptl.attach_label(t, full, pair, False)
    dq, dr = t[t["event_type"] == "DEQUEUE"], t[t["event_type"] == "DROP"]
    dur = max(dq["Timestamp"].max() - dq["Timestamp"].min(), 1e-9)
    rows = []
    for c in ["All"] + CLASSES:
        q = dq if c == "All" else dq[dq["cls"] == c]
        d = dr if c == "All" else dr[dr["cls"] == c]
        n = len(q) + len(d)
        rows.append(dict(cls=c, n_pkts=n, n_drop=len(d),
                         delay_ms=q["d_queue_ms"].mean() if len(q) else np.nan,
                         p95_delay_ms=q["d_queue_ms"].quantile(.95) if len(q) else np.nan,
                         loss=len(d) / n if n else np.nan,
                         thr_mbps=q["payload_size"].sum() * 8 / 1e6 / dur))
    return rows


def run(a):
    import eval_dcod_multiseed as ev
    import dcod_env
    import simulate_dcodel as sd
    from dcod_env import DCoDEnv
    import run_dn_dcod_labeled as rl

    resolve_out(a)
    if a.policy_dir:
        ev.RESULTS_DQN = a.policy_dir
    if a.cpu:   # mang rat nho, batch=1: CPU nhanh hon GPU (khong phai chuyen du lieu qua GPU moi goi)
        import torch
        import train_dqn_dcod as tdd
        tdd.DEVICE = "cpu"
        torch.set_num_threads(1)
        print("[cpu] suy luan chinh sach tren CPU")
    shield_ms = a.shield_ms if a.shield_ms is not None else float(dcod_env.T_MAX_TARGET_MS)
    random.seed(0); np.random.seed(0)
    tmp = a.tmp_dir
    os.makedirs(tmp, exist_ok=True)

    done = set()
    res = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    if len(res):
        done = set(zip(res["rate"], res["seed"], res["aqm"]))
    agent_none = ev.load_policy(a.job_none, "none")
    agent_label = ev.load_policy(a.job_label, "external")

    for s in a.seeds:
        if s == ev.TRAIN_SEED:
            print(f"[seed {s}] la vet huan luyen -> bo qua"); continue
        dt = ev.trace("droptail", s, a.run)
        lp = ev.labels_path(s, a.run)
        if not os.path.exists(dt) or not os.path.exists(lp):
            print(f"[seed {s}] thieu {dt} hoac {lp} -> bo qua"); continue
        full, pair = ptl.load_labels(lp)
        for rate in a.rates:
            dcod_env.BOTTLENECK_RATE_MBPS = float(rate)
            sd.BOTTLENECK_RATE_MBPS = float(rate)
            for aqm in a.methods:
                if (float(rate), s, aqm) in done:
                    continue
                t0 = time.time()
                reuse = (abs(rate - BASE_RATE) < 1e-9 and aqm in ("dcodel", "dn_dcod_khong_nhan", "dn_dcod"))
                out = ev.trace(aqm, s, a.run) if reuse else os.path.join(tmp, f"{aqm}_s{s}_r{rate}.csv")
                if reuse and not os.path.exists(out):
                    print(f"[seed {s}] thieu vet co san {out} -> bo qua"); continue
                if not reuse:
                    if aqm == "dcodel":
                        sd.run_dcodel_simulation(dt, out)
                    elif aqm == "keep_all":
                        DCoDEnv(dt, direction=0, label_mode="none").run_greedy_and_export(KeepAgent(), out)
                    elif aqm == "keep_shield":   # DOI CHUNG: khong DQN, chi KEEP + ep DROP khi sojourn > nguong
                        ag = ShieldAgent(KeepAgent(), shield_ms)
                        DCoDEnv(dt, direction=0, label_mode="none").run_greedy_and_export(ag, out)
                        print(f"    [shield] ep DROP {ag.n_over}/{ag.n_all} goi ({100*ag.n_over/max(ag.n_all,1):.2f}%) khi sojourn > {ag.thr:g} ms")
                    else:
                        use_label = aqm in ("dn_dcod", "dn_dcod_shield", "dn_dcod_online")
                        ag = agent_label if use_label else agent_none
                        if aqm.endswith("_shield"):
                            ag = ShieldAgent(ag, shield_ms)
                        env = DCoDEnv(dt, direction=0, label_mode="external" if use_label else "none")
                        if use_label:
                            env.set_label_provider(rl.build_label_provider(lp))
                        if aqm.endswith("_online"):
                            import torch
                            from online_adapt import OnlineAgent
                            random.seed(s); np.random.seed(s); torch.manual_seed(s)
                            ag = OnlineAgent(ev.load_policy(a.job_label if use_label else a.job_none,
                                                            "external" if use_label else "none"),
                                             eps=a.online_eps, lr=a.online_lr, update_every=a.online_every,
                                             window=a.online_window, margin=a.online_margin,
                                             rollback=not a.no_rollback)
                            ag.attach(env)
                        env.run_greedy_and_export(ag, out)
                        if hasattr(ag, "summary"):
                            print("    " + ag.summary())
                        if isinstance(ag, ShieldAgent):
                            print(f"    [shield] ep DROP {ag.n_over}/{ag.n_all} goi ({100*ag.n_over/max(ag.n_all,1):.2f}%) khi sojourn > {ag.thr:g} ms")
                rows = metrics_from_trace(out, full, pair)
                if not rows or rows[0]["n_pkts"] == 0 or not np.isfinite(rows[0]["delay_ms"]):
                    print(f"[seed {s} | {rate:g} Mbps | {aqm}] vet khong co DEQUEUE hop le ({out}) -> bo qua, KHONG luu")
                    if not reuse and os.path.exists(out):
                        os.remove(out)
                    continue
                for r in rows:
                    r.update(rate=float(rate), seed=s, aqm=aqm)
                res = pd.concat([res, pd.DataFrame(rows)], ignore_index=True)
                res.to_csv(OUT_CSV, index=False)
                done.add((float(rate), s, aqm))
                if not reuse and os.path.exists(out):
                    os.remove(out)
                r0 = rows[0]
                print(f"[seed {s:>2} | {rate:>4} Mbps | {aqm:20s}] delay={r0['delay_ms']:.2f} ms  "
                      f"loss={100*r0['loss']:.2f}%  thr={r0['thr_mbps']:.3f}  ({time.time()-t0:.0f}s)")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nXong. Ket qua: {OUT_CSV}.  Buoc tiep: --stage report")


def report(a):
    from scipy.stats import wilcoxon
    import matplotlib.pyplot as plt
    resolve_out(a)
    res = pd.read_csv(OUT_CSV)
    rates = sorted(res["rate"].unique(), reverse=True)
    LOWER = {"delay_ms": True, "p95_delay_ms": True, "loss": True, "thr_mbps": False}
    PRIMARY = {("Mice", "delay_ms"), ("Elephant", "loss"), ("Elephant", "thr_mbps")}
    # moi "ho" gom 3 phuong phap & 3 cap dinh truoc: (nhan vs dCoDel), (nhan vs khong nhan), (khong nhan vs dCoDel)
    FAMILIES = [("A. chinh sach DQN dong bang", "dcodel", "dn_dcod_khong_nhan", "dn_dcod"),
                ("B. chinh sach DQN + bo bao ve", "dcodel", "dn_dcod_khong_nhan_shield", "dn_dcod_shield"),
                ("D. chinh sach DQN tu dieu chinh khi chay", "dcodel", "dn_dcod_khong_nhan_online", "dn_dcod_online")]
    # cap bo sung (tac dung cua bo bao ve, tung bien the): ho C, 3 cap
    EXTRAS = [("C. tac dung cua bo bao ve", [("dn_dcod_shield", "dn_dcod"), ("dn_dcod_khong_nhan_shield", "dn_dcod_khong_nhan"),
                                             ("dn_dcod_shield", "dn_dcod_khong_nhan_shield")]),
              ("F. doi chung chi bo bao ve (khong DQN)", [("dn_dcod_shield", "keep_shield"),
                                             ("dn_dcod_khong_nhan_shield", "keep_shield"), ("keep_shield", "dcodel")]),
              ("E. tac dung cua tu dieu chinh (so voi dong bang)", [("dn_dcod_online", "dn_dcod"),
                                             ("dn_dcod_khong_nhan_online", "dn_dcod_khong_nhan"),
                                             ("dn_dcod_online", "dn_dcod_khong_nhan_online")])]

    def families():
        for name, dc, nn, lb in FAMILIES:
            yield name, [dc, nn, lb], [(lb, dc), (lb, nn), (nn, dc)]
        for nm, prs in EXTRAS:
            yield nm, sorted({k for p in prs for k in p}), prs

    print("\n=== So seed hoan chinh theo toc do ===")
    for r in rates:
        sub = res[(res.rate == r) & (res.cls == "All")].pivot(index="seed", columns="aqm", values="loss")
        for name, cols, _ in families():
            if all(c in sub for c in cols):
                print(f"  {r:g} Mbps | {name}: {sub.dropna(subset=cols).shape[0]} seed")

    for name, cols, PAIRS in families():
        print(f"\n################ Ho {name} ################")
        for cls, metrics in [("All", ["delay_ms", "p95_delay_ms", "loss", "thr_mbps"]),
                             ("Mice", ["delay_ms", "loss", "thr_mbps"]), ("Sheep", ["loss", "thr_mbps"]),
                             ("Elephant", ["loss", "thr_mbps"])]:
            print(f"\n######## Lop: {cls} ########")
            for r in rates:
                for m in metrics:
                    w = res[(res.rate == r) & (res.cls == cls)].pivot(index="seed", columns="aqm", values=m)
                    if not all(c in w for c in cols):
                        continue
                    w = w.dropna(subset=cols)
                    if len(w) < 7:
                        continue
                    tag = "(*)" if (cls, m) in PRIMARY else "   "
                    ms = "  ".join(f"{k}={w[k].mean():.4f}±{w[k].std():.4f}" for k in cols)
                    print(f"{tag} {r:g} Mbps {m:13s} {ms}")
                    ps, info = [], []
                    for x, y in PAIRS:
                        d = w[x] - w[y]
                        better = (d < 0) if LOWER[m] else (d > 0)
                        p = 1.0 if (d == 0).all() else wilcoxon(w[x], w[y], method="exact").pvalue
                        ps.append(p); info.append((x, y, d.mean(), int(better.sum()), len(w)))
                    for (x, y, dm, nb, n), p, q in zip(info, ps, ptl.holm(ps)):
                        print(f"        {x} vs {y}: hieu TB={dm:+.4f}, {x} tot hon {nb}/{n} seed, p_Holm={q:.4f}")

    # --- Kiem chung tren seed doc lap: cac chenh lech chinh co cung chieu o seed moi khong? ---
    if a.confirm_seeds:
        CONF = [("Mice", "loss", "dn_dcod_shield", "dcodel"), ("Sheep", "loss", "dn_dcod_shield", "dcodel"),
                ("Elephant", "loss", "dn_dcod_shield", "dcodel"), ("Mice", "thr_mbps", "dn_dcod_shield", "dcodel"),
                ("Sheep", "thr_mbps", "dn_dcod_shield", "dcodel"), ("Elephant", "thr_mbps", "dn_dcod_shield", "dcodel"),
                ("Elephant", "thr_mbps", "dn_dcod_shield", "dn_dcod_khong_nhan_shield"),
                ("All", "delay_ms", "dn_dcod_shield", "dn_dcod_khong_nhan_shield"),
                ("All", "delay_ms", "dn_dcod_shield", "dn_dcod"), ("All", "p95_delay_ms", "dn_dcod_shield", "dn_dcod")]
        cs = set(a.confirm_seeds)
        print(f"\n################ KIEM CHUNG tren seed doc lap {sorted(cs)} ################")
        print("Chieu ky vong = dau cua chenh lech TB tren cac seed con lai (tham chieu). 'cung chieu' = so seed moi trung chieu.")
        for r in rates:
            for cls, m, x, y in CONF:
                w = res[(res.rate == r) & (res.cls == cls)].pivot(index="seed", columns="aqm", values=m)
                if x not in w or y not in w:
                    continue
                d = (w[x] - w[y]).dropna()
                ref, new_ = d[~d.index.isin(cs)], d[d.index.isin(cs)]
                if len(ref) < 5 or len(new_) == 0:
                    continue
                sg = np.sign(ref.mean())
                same = int((np.sign(new_) == sg).sum())
                print(f"  {r:g} Mbps {cls:8s} {m:13s} {x} - {y}: tham chieu TB={ref.mean():+.4f} (n={len(ref)}) | "
                      f"seed moi: " + ", ".join(f"{i}:{v:+.4f}" for i, v in new_.items()) + f" -> cung chieu {same}/{len(new_)}")

    # --- Bieu do (chi hien thi, khong luu) ---
    COL = {"keep_all": "#7F8C8D", "keep_shield": "#7F8C8D", "dcodel": "#2E86C1", "dn_dcod_khong_nhan": "#E67E22", "dn_dcod": "#C0392B",
           "dn_dcod_khong_nhan_shield": "#E67E22", "dn_dcod_shield": "#C0392B",
           "dn_dcod_khong_nhan_online": "#E67E22", "dn_dcod_online": "#C0392B"}
    NAME = {"keep_all": "KEEP-all (DropTail)", "keep_shield": "Chỉ bộ bảo vệ (không DQN)", "dcodel": "dCoDel", "dn_dcod_khong_nhan": "DN-dCoD (không nhãn)",
            "dn_dcod": "DN-dCoD (nhúng nhãn)",
            "dn_dcod_khong_nhan_shield": "DN-dCoD (không nhãn) + bảo vệ", "dn_dcod_shield": "DN-dCoD (nhúng nhãn) + bảo vệ",
            "dn_dcod_khong_nhan_online": "DN-dCoD (không nhãn) tự điều chỉnh", "dn_dcod_online": "DN-dCoD (nhúng nhãn) tự điều chỉnh"}

    def panel(ax, cls, m, ttl, scale=1.0):
        for k in [k for k in AQMS if k in set(res["aqm"])]:
            mu, sd_ = [], []
            for r in rates:
                v = res[(res.rate == r) & (res.cls == cls) & (res.aqm == k)][m] * scale
                mu.append(v.mean()); sd_.append(v.std())
            mu, sd_ = np.array(mu), np.array(sd_)
            ax.plot(rates, mu, ("--s" if k.endswith("_shield") else ":^" if k.endswith("_online") else "-o"), color=COL[k], lw=2.2, ms=5, label=NAME[k])
            ax.fill_between(rates, mu - sd_, mu + sd_, color=COL[k], alpha=.15, lw=0)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_xlabel("Tốc độ nút cổ chai (Mbps)  —  tắc nghẽn tăng →")
        ax.invert_xaxis(); ax.grid(alpha=.3)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    panel(ax[0, 0], "All", "delay_ms", "Độ trễ TB (ms)")
    panel(ax[0, 1], "All", "loss", "Tỷ lệ mất gói (%)", 100)
    panel(ax[1, 0], "All", "thr_mbps", "Thông lượng (Mbps)")
    panel(ax[1, 1], "All", "p95_delay_ms", "Độ trễ p95 (ms)")
    ax[0, 0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Kiểm thử đa tải (10 seed, trung bình ± SD)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96)); plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    panel(ax[0], "Mice", "delay_ms", "Mice – độ trễ TB (ms)")
    panel(ax[1], "Elephant", "loss", "Elephant – mất gói (%)", 100)
    panel(ax[2], "Elephant", "thr_mbps", "Elephant – thông lượng (Mbps)")
    ax[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Theo lớp luồng (chỉ số định trước)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94)); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "report"], default="run")
    ap.add_argument("--rates", type=float, nargs="+", default=[10.0, 8.0, 6.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--job_none", default="dn_dcod_khong_nhan")
    ap.add_argument("--job_label", default="dn_dcod")
    ap.add_argument("--policy_dir", default=None)
    ap.add_argument("--tmp_dir", default="/tmp/da_tai")
    ap.add_argument("--out", default=None, help="duong dan da_tai_metrics.csv (mac dinh: <policy_dir>/da_tai_metrics.csv)")
    ap.add_argument("--cpu", action="store_true", help="suy luan DQN tren CPU (thuong nhanh hon GPU o batch=1)")
    ap.add_argument("--methods", nargs="+", default=AQMS_BASE, choices=AQMS,
                    help="chon phuong phap (mac dinh 4 phuong phap goc); *_shield = them bo bao ve; keep_all chi la moc mo ta")
    ap.add_argument("--confirm_seeds", type=int, nargs="+", default=None,
                    help="(report) seed doc lap de kiem chung chieu cua cac chenh lech chinh (vd 8 9 11)")
    ap.add_argument("--online_eps", type=float, default=0.01, help="(online) xac suat hanh dong ngau nhien")
    ap.add_argument("--online_lr", type=float, default=1e-4, help="(online) toc do hoc")
    ap.add_argument("--online_every", type=int, default=4, help="(online) cap nhat mang moi N goi")
    ap.add_argument("--online_window", type=int, default=5000, help="(online) kich thuoc cua so (goi) cho phuong an du phong")
    ap.add_argument("--online_margin", type=float, default=0.25, help="(online) tut phan thuong toi da (ty le) truoc khi rollback")
    ap.add_argument("--no_rollback", action="store_true", help="(online) tat phuong an du phong (de doi chung)")
    ap.add_argument("--shield_ms", type=float, default=None,
                    help="nguong sojourn de ep DROP (mac dinh = T_MAX_TARGET_MS cua env, KHONG chinh theo du lieu thu)")
    a = ap.parse_args()
    run(a) if a.stage == "run" else report(a)
