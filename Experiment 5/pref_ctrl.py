#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIEU KHIEN THONG MINH BANG THAM SO UU TIEN (reward) cho DN-dCoD NHUNG NHAN FRCNC (NS-3 va UKM).
KHONG thay chinh sach Chuong 4 (giu nguyen ket qua o 10 Mbps); day la bien the bo sung.

Y tuong: 1 "nut" lambda in [0.1, 0.9] noi muc uu tien chong tre:
    reward = -( a*beta1*D/Dmax + b*(beta2*Loss + beta3*(1-Thr/C)) ),  a = 2*lambda, b = 2*(1-lambda)
    (beta1..3 van theo nhan Mice/Sheep/Elephant nhu Chuong 3; lambda=0.5 -> dung reward goc)
DQN duoc huan luyen CO DIEU KIEN theo lambda (lambda la 1 dac trung cua state; doi ngau nhien moi 1000 goi,
nhieu muc tai: NS-3 6/8/10 Mbps, doan ngau nhien 5000 goi; UKM doan nghen ngau nhien).
Luc chay, 1 BO DIEU KHIEN DINH TRUOC (chi dung tin hieu quan sat duoc khi chay) chinh lambda:
    moi W=1000 goi: tre_uoc_tinh_TB > D_REF  -> lambda += 0.1 ;  < 0.5*D_REF -> lambda -= 0.1 ;  kep [0.1,0.9]
    D_REF = 0.1 * D_MAX  (NS-3: 20 ms; UKM: 100 ms).   KHONG chinh theo du lieu thu.
Danh gia: lambda co dinh {0.2,0.5,0.8} (nut co tac dung that khong?) va bo dieu khien (pref_ctrl).

Colab (D, bien moi truong, %cd nhu moi lan):
  NS-3:  !python3 pref_ctrl.py --data ns3 --stage train --cpu
         !python3 pref_ctrl.py --data ns3 --stage run --cpu --seeds 1 2 3 4 5
         !python3 pref_ctrl.py --data ns3 --stage report
  UKM:   !python3 pref_ctrl.py --data ukm --stage train --cpu
         !python3 pref_ctrl.py --data ukm --stage run --cpu
         !python3 pref_ctrl.py --data ukm --stage report
"""
import os, sys, time, json, random, argparse
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", ".")
for p in (DATA_DIR, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
import dcod_env
from dcod_env import DCoDEnv

LAM_LO, LAM_HI, LAM_SWITCH = 0.1, 0.9, 1000
FIXED = {"pref_fix20": 0.2, "pref_fix50": 0.5, "pref_fix80": 0.8}
AQMS = list(FIXED) + ["pref_ctrl", "pref_ctrl2", "pref_ctrl3"]
CTRL_W, CTRL_STEP, D_REF_FRAC = 1000, 0.1, 0.1
C3_WARM, C3_K, C3_STEP, C3_LO, C3_HI = 5, 1.5, 0.05, 0.4, 0.6      # ctrl3: chot truoc, khong chinh sau khi xem ket qua
TRAIN_RATES = [6.0, 8.0, 10.0]
SEG = {"ns3": 5000, "ukm": 6000}
TRAIN_SEED, RUN = 10, 1
DELAY_POW = 0.5          # v2 (pref_ctrl_v1.py: 1.0)
JOB = {"none": "pref_khong_nhan", "external": "pref_dn_dcod"}
# --- bien the ukm2 (chi UKM): (i) so hang tre = tre UOC TINH cua hang doi SAU hanh dong (tin hieu dan dau, tuc thoi),
# thay cho tre cua goi da roi hang doi (tre pha, bi tri hoan hang tram goi nen DROP khong duoc "ghi cong");
# (ii) dang loi 2 tham so: x=est/EST_S; x^2 neu x<=2, sau do tuyen tinh doc EST_SLOPE=2 (khong bao hoa toi 1200 ms = day hang doi 2 Mbps), tran x=EST_X_CAP; nhan EST_SCALE;
# (iii) chi 2 hanh dong KEEP/DROP (T+/T- khong tac dung len drop, chi lam loang kham pha va thien vi Q cua KEEP).
EST_S, EST_X_CAP, EST_SCALE, EST_SLOPE = 100.0, 12.0, 0.25, 2.0


class PrefMixin:
    lam = 0.5
    lam0 = 0.5

    def _current_state(self):
        return np.append(super()._current_state(), self.lam).astype(np.float32)

    def _reward(self, beta):
        """Giong DCoDEnv._reward (3.48) nhung (i) beta nhan voi a=2*lam / b=2*(1-lam); (ii) so hang tre dung
        (D/Dmax)^DELAY_POW (v1: pow=1; v2: 0.5 de tre vai chuc ms van co tin hieu hoc, khong bi chim boi mat goi)."""
        a, b = 2.0 * self.lam, 2.0 * (1.0 - self.lam)
        if getattr(self, "est_reward", False):
            est = sum(p["payload_size"] for p in self.queue) * 8 / self.rate_bps * 1000.0
            x = min(est / EST_S, EST_X_CAP)
            dterm = EST_SCALE * (x * x if x <= 2.0 else 4.0 + EST_SLOPE * (x - 2.0))
            d_avg_dummy = None
        else:
            d_avg_dummy = True
        d_avg = float(np.mean(self.recent_delays_ms)) if self.recent_delays_ms else 0.0
        loss_rate = float(np.mean(self.recent_drops)) if self.recent_drops else 0.0
        if len(self.recent_times_s) >= 2:
            span = self.recent_times_s[-1] - self.recent_times_s[0]
            thr = (sum(self.recent_bytes) * 8 / 1e6) / span if span > 1e-6 else 0.0
        else:
            thr = 0.0
        if d_avg_dummy:
            dterm = min(d_avg / dcod_env.D_MAX_MS, 1.0) ** DELAY_POW
        return -(a * beta["beta1"] * dterm + b * (beta["beta2"] * loss_rate
                 + beta["beta3"] * (1.0 - min(thr / dcod_env.BOTTLENECK_RATE_MBPS, 1.0))))


class NS3Env(PrefMixin, DCoDEnv):
    def __init__(self, csv, label_mode="external"):
        super().__init__(csv, direction=0, label_mode=label_mode)
        self._full = self.arrivals
        self.state_dim += 1
        self.rate_mbps = dcod_env.BOTTLENECK_RATE_MBPS

    def reset(self, rate=None, start=0, n=None):
        if rate is not None:
            self.rate_mbps = float(rate)
        dcod_env.BOTTLENECK_RATE_MBPS = self.rate_mbps
        self.rate_bps = self.rate_mbps * 1e6
        a = self._full.iloc[start:start + n] if n else self._full
        self.arrivals = a.reset_index(drop=True)
        self.n = len(self.arrivals)
        self.lam = self.lam0
        return super().reset()


def make_ukm_env(label_mode, rate_mbps, label_fn, variant=""):
    import run_ukm_dcod as rk
    env = rk.make_env(label_mode, rate_mbps, label_fn)
    env.__class__ = type("PrefUKM", (PrefMixin, env.__class__), {})
    env.state_dim += 1
    env.lam = env.lam0 = 0.5
    if variant == "ukm2":
        env.est_reward = True
        env.action_dim = 2
    return env


class CtrlAgent:
    """Boc DQN: ghi lambda vao state; che do co dinh hoac bo dieu khien dinh truoc."""
    def __init__(self, base, env, lam0=0.5, adaptive=False, v2=False, d_ref_ms=None, v3=False):
        self.base, self.env, self.lam0, self.adaptive, self.v2, self.v3 = base, env, lam0, adaptive, v2, v3
        self.hist = []            # ctrl3: tre tung cua so de TU HIEU CHUAN D_ref
        self.d_ref_ms = d_ref_ms        # None: D_REF_FRAC*D_MAX (mac dinh); so: nguong tuyet doi (vd 20 ms = gia tri NS-3 khi chuyen sang UKM)
        self.n, self.acc, self.lams, self.quiet = 0, 0.0, [], 0

    def select_action(self, s, greedy=True):
        env = self.env
        if self.n == 0:
            env.lam = self.lam0
        if self.adaptive and self.n > 0 and self.n % CTRL_W == 0:
            d_ref = self.d_ref_ms if self.d_ref_ms else D_REF_FRAC * dcod_env.D_MAX_MS
            obs = self.acc / CTRL_W
            if self.v3:
                # ctrl3 (TU DONG: khong hang so theo mang). lambda0 = 0,5; D_ref = 1,5 x trung vi tre 5 cua so dau (tu hieu chuan);
                # tre > D_ref -> +C3_STEP; tre < 0,5*D_ref lien tiep 10 cua so -> -C3_STEP; ket lambda trong [0,4; 0,6] (khong qua "vach" 0,5-0,8).
                self.hist.append(obs)
                if len(self.hist) > C3_WARM:
                    d_ref = C3_K * float(np.median(self.hist[:C3_WARM]))
                    if obs > d_ref:
                        env.lam = min(C3_HI, env.lam + C3_STEP); self.quiet = 0
                    elif obs < 0.5 * d_ref:
                        self.quiet += 1
                        if self.quiet >= 10:
                            env.lam = max(C3_LO, env.lam - C3_STEP); self.quiet = 0
                    else:
                        self.quiet = 0
            elif self.v2:
                # ctrl2 (dinh truoc sau khi thay "vach" lambda 0.5->0.8 o seed phat trien 1): bat dau lambda=0.8; tre > D_REF -> +0.1 ngay
                # (tran 0.9); chi ha -0.1 sau 10 cua so lien tiep tre < 0.5*D_REF; san 0.6 (khong xuong vung chinh sach it rot).
                if obs > d_ref:
                    env.lam = min(LAM_HI, env.lam + CTRL_STEP); self.quiet = 0
                elif obs < 0.5 * d_ref:
                    self.quiet += 1
                    if self.quiet >= 10:
                        env.lam = max(0.6, env.lam - CTRL_STEP); self.quiet = 0
                else:
                    self.quiet = 0
            elif obs > d_ref:
                env.lam = min(LAM_HI, env.lam + CTRL_STEP)
            elif obs < 0.5 * d_ref:
                env.lam = max(LAM_LO, env.lam - CTRL_STEP)
            self.acc = 0.0
        self.acc += float(s[0]) * 100.0
        self.n += 1
        self.lams.append(env.lam)
        s2 = np.array(s, dtype=np.float32)
        s2[-1] = env.lam
        return self.base.select_action(s2, greedy=True)

    def mean_lam(self):
        return float(np.mean(self.lams)) if self.lams else float("nan")


def load_dqn(path):
    import torch
    import train_dqn_dcod as tdd
    ck = torch.load(path, map_location=tdd.DEVICE)
    ag = tdd.DQNAgent(ck["state_dim"], ck["action_dim"])
    ag.q_net.load_state_dict(ck["model_state"]); ag.q_net.eval()
    return ag


MICE_BETA = dict(beta1=1.2, beta2=1.2, beta3=0.6)   # bien the "mice": rot Mice bi phat nang hon (goc: 1.8/0.6/0.6), tong van = 3.0


def out_dir(a):
    if a.out_dir is None and getattr(a, "variant", "") == "mice":
        return os.path.join(DATA_DIR, "results" if a.data == "ns3" else "results_ukm", "pref_mice")
    if a.out_dir is None and getattr(a, "variant", "") == "ukm2":
        assert a.data == "ukm", "--variant ukm2 chi cho --data ukm"
        return os.path.join(DATA_DIR, "results_ukm", "pref_ukm2")
    return a.out_dir or (os.path.join(DATA_DIR, "results", "pref2") if a.data == "ns3"
                         else os.path.join(DATA_DIR, "results_ukm", "pref2"))


def train(a):
    import torch
    import train_dqn_dcod as tdd
    import run_dn_dcod_labeled as rl
    if a.cpu:
        tdd.DEVICE = "cpu"; torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    mode, job, od = a.label_mode, JOB[a.label_mode], out_dir(a)
    os.makedirs(od, exist_ok=True)
    if getattr(a, "variant", "") == "mice":
        dcod_env.BETA_BY_LABEL["Mice"] = dict(MICE_BETA)
        print(f"[beta] Mice = {MICE_BETA} (goc 1.8/0.6/0.6)")
    steps = SEG[a.data]
    if a.data == "ns3":
        import eval_dcod_multiseed as ev
        env = NS3Env(ev.trace("droptail", TRAIN_SEED, RUN), mode)
        if mode == "external":
            env.set_label_provider(rl.build_label_provider(ev.labels_path(TRAIN_SEED, RUN)))
        maxstart = max(1, len(env._full) - steps)
    else:
        import run_ukm_dcod as rk
        rk.UKM_RUN = 0
        fn = rl.build_label_provider(rk.labels_csv()) if mode == "external" else None
        env = make_ukm_env(mode, a.ukm_rate, fn, getattr(a, "variant", ""))
        env.seg_len = steps
        env.set_congested_starts(a.ukm_rate)
    tdd.EPS_DECAY_STEPS = int(0.4 * a.episodes * steps)
    ag = tdd.DQNAgent(env.state_dim, env.action_dim)
    ck_path = os.path.join(od, f"{job}_ckpt.pt")
    ep0, log = 0, []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=tdd.DEVICE)
        ag.q_net.load_state_dict(ck["q"]); ag.target_net.load_state_dict(ck["t"])
        ag.opt.load_state_dict(ck["o"]); ag.global_step = ck["g"]; ep0 = ck["ep"] + 1; log = ck["log"]
        print(f"[{job}] tiep tuc tu episode {ep0}")
    for ep in range(ep0, a.episodes):
        t0 = time.time()
        rng = np.random.RandomState(1000 + ep)
        if a.data == "ns3":
            rate = TRAIN_RATES[ep % 3]
            s = env.reset(rate=rate, start=int(rng.randint(0, maxstart)), n=steps)
        else:
            rate = a.ukm_rate
            s = env.reset()
        R, lam_log, n_dact = 0.0, [], 0
        for t in range(min(steps, env.n)):
            if t % LAM_SWITCH == 0:
                env.lam = float(rng.uniform(LAM_LO, LAM_HI)); lam_log.append(env.lam)
                s = env._current_state()
            act = ag.select_action(s)
            n_dact += int(act == 1)
            s2, r, done, _ = env.step(act)
            ag.store(s, act, r, s2, done)
            ag.train_step()
            s, R = s2, R + r
            if done: break
        sm = env.episode_summary()
        log.append(dict(ep=ep + 1, rate=rate, reward=R, delay=sm["avg_delay_ms"], loss=sm["loss_rate"],
                        lam_mean=float(np.mean(lam_log))))
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"[{job}] ep {ep+1}/{a.episodes} rate={rate:g} lam~{log[-1]['lam_mean']:.2f} reward={R:.1f} "
                  f"eps={ag.epsilon():.2f} delay={sm['avg_delay_ms']:.1f}ms loss={100*sm['loss_rate']:.1f}% "
                  f"DROP-chu-dong={100*n_dact/max(1,t+1):.1f}% ({time.time()-t0:.0f}s)")
        torch.save(dict(q=ag.q_net.state_dict(), t=ag.target_net.state_dict(), o=ag.opt.state_dict(),
                        g=ag.global_step, ep=ep, log=log), ck_path)
    torch.save(dict(model_state=ag.q_net.state_dict(), state_dim=env.state_dim, action_dim=env.action_dim,
                    label_mode=mode), os.path.join(od, f"{job}_qnet.pt"))
    json.dump(log, open(os.path.join(od, f"{job}_log.json"), "w"))
    print(f"[{job}] xong -> {od}/{job}_qnet.pt (checkpoint CUOI)")


def _agent_for(aqm, base, env, d_ref_ms=None):
    if aqm == "pref_ctrl":
        return CtrlAgent(base, env, 0.5, adaptive=True, d_ref_ms=d_ref_ms)
    if aqm == "pref_ctrl3":
        return CtrlAgent(base, env, 0.5, adaptive=True, v3=True)
    if aqm == "pref_ctrl2":
        return CtrlAgent(base, env, 0.8, adaptive=True, v2=True, d_ref_ms=d_ref_ms)
    return CtrlAgent(base, env, FIXED[aqm], adaptive=False)


def run(a):
    import train_dqn_dcod as tdd
    import run_dn_dcod_labeled as rl
    import phan_tich_theo_lop as ptl
    if a.cpu:
        import torch; tdd.DEVICE = "cpu"; torch.set_num_threads(1)
    od = out_dir(a); os.makedirs(od, exist_ok=True)
    tag = a.tag if a.tag is not None else (f"_shield{a.shield_ms:g}" if a.shield_ms else "")
    csv = os.path.join(od, f"pref_metrics{tag}.csv")
    res = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    done = set(zip(res["rate"], res["seed"], res["aqm"])) if len(res) else set()
    base = load_dqn(os.path.join(getattr(a, "model_dir", None) or od, f"{JOB['external']}_qnet.pt"))
    os.makedirs(a.tmp_dir, exist_ok=True)
    if a.data == "ns3":
        import eval_dcod_multiseed as ev
        import eval_da_tai as eda
        jobs = [(s, r) for s in a.seeds for r in a.rates]
    else:
        import run_ukm_dcod as rk
        import ukm_mo_rong as um
        jobs = [(s, a.ukm_rate) for s in a.runs]
    for s, rate in jobs:
        if a.data == "ns3":
            if s == TRAIN_SEED: continue
            dt, lp = ev.trace("droptail", s, RUN), ev.labels_path(s, RUN)
        else:
            rk.UKM_RUN = s; lp = rk.labels_csv()
            if not (os.path.exists(rk.trace_csv()) and os.path.exists(lp)):
                print(f"[run {s}] thieu vet/nhan -> bo qua"); continue
        if a.data == "ns3" and not (os.path.exists(dt) and os.path.exists(lp)):
            print(f"[seed {s}] thieu vet/nhan -> bo qua"); continue
        full, pair = ptl.load_labels(lp)
        for aqm in (a.aqms or AQMS):
            if (float(rate), s, aqm) in done: continue
            t0 = time.time()
            fn = rl.build_label_provider(lp)
            if a.data == "ns3":
                dcod_env.BOTTLENECK_RATE_MBPS = float(rate)
                env = NS3Env(dt, "external"); env.set_label_provider(fn); env.reset(rate=rate)
            else:
                env = make_ukm_env("external", rate, fn, getattr(a, "variant", ""))
            ctrl = _agent_for(aqm, base, env, getattr(a, "d_ref_ms", None))
            ag = ctrl
            if a.shield_ms:
                import eval_da_tai as eda2
                ag = eda2.ShieldAgent(ctrl, a.shield_ms)
            out = os.path.join(a.tmp_dir, f"{aqm}_s{s}_r{rate}.csv")
            env.run_greedy_and_export(ag, out)
            if a.data == "ns3":
                rows = eda.metrics_from_trace(out, full, pair)
            else:
                df, _ = um.per_class_all(ptl, out, full, pair); rows = df.to_dict("records")
            os.remove(out)
            if not rows or not np.isfinite(rows[0]["delay_ms"]):
                print(f"[{s} | {rate:g} | {aqm}] vet khong hop le -> bo qua"); continue
            for r in rows:
                r.update(rate=float(rate), seed=s, aqm=aqm, lam_mean=ctrl.mean_lam())
            res = pd.concat([res, pd.DataFrame(rows)], ignore_index=True)
            res.to_csv(csv, index=False); done.add((float(rate), s, aqm))
            r0 = [r for r in rows if r["cls"] == "All"][0]
            print(f"[{s:>2} | {rate:>4g} Mbps | {aqm:10s}] delay={r0['delay_ms']:.2f} ms loss={100*r0['loss']:.2f}% "
                  f"thr={r0['thr_mbps']:.3f} lam_TB={ctrl.mean_lam():.2f} ({time.time()-t0:.0f}s)")
    print(f"\nXong -> {csv}.  Buoc tiep: --stage report")


def report(a):
    from scipy.stats import wilcoxon
    od = out_dir(a)
    tag = a.tag if a.tag is not None else (f"_shield{a.shield_ms:g}" if a.shield_ms else "")
    cur = pd.read_csv(os.path.join(od, f"pref_metrics{tag}.csv"))
    base = pd.read_csv(a.base_csv) if a.base_csv and os.path.exists(a.base_csv) else pd.DataFrame()
    if len(base) and "rate" not in base:
        base["rate"] = a.ukm_rate
    keep = ["dcodel", "dn_dcod", "dn_dcod_shield", "keep_shield"]
    parts = [cur, base[base.aqm.isin(keep)] if len(base) else base]
    if a.extra_csv and os.path.exists(a.extra_csv):      # vd. da_tai_shield10.csv: doi chung nguong co dinh, KHONG DQN, KHONG nhan
        ex = pd.read_csv(a.extra_csv); ex = ex[ex.aqm == "keep_shield"].copy(); ex["aqm"] = a.extra_name
        if "rate" not in ex:
            ex["rate"] = a.ukm_rate      # UKM: file doi chung khong co cot rate (truoc day thanh nhom "nan Mbps")
        parts.append(ex)
    res = pd.concat(parts, ignore_index=True)
    pairs = [("pref_fix80", "pref_fix50"), ("pref_fix80", "dcodel"), ("pref_fix80", "keep_shield"), ("pref_fix80", "dn_dcod"),
             ("pref_ctrl2", "pref_fix80"), ("pref_ctrl2", "dcodel"), ("pref_ctrl2", "keep_shield"), ("pref_ctrl2", "dn_dcod"),
             ("pref_fix50", "keep_shield"), ("pref_fix50", "dn_dcod"), ("pref_fix50", "dcodel"),
             ("pref_fix80", a.extra_name), ("pref_ctrl2", a.extra_name)]
    for r in sorted(res.rate.unique(), reverse=True):
        print(f"\n=== {r:g} Mbps (All) ===")
        for m, lab in (("delay_ms", "do tre ms"), ("loss", "mat goi"), ("thr_mbps", "thong luong")):
            w = res[(res.rate == r) & (res.cls == "All")].pivot(index="seed", columns="aqm", values=m)
            print(f"  {lab}: " + " | ".join(f"{k} {w[k].mean():.3f}(n={w[k].count()})" for k in w.columns))
            ps = []
            for x, y in pairs:
                if x in w and y in w:
                    d = (w[x] - w[y]).dropna()
                    if len(d) >= 5 and (d != 0).any():
                        ps.append((x, y, d.mean(), int((d < 0).sum()), len(d), wilcoxon(d).pvalue))
            order = np.argsort([p[5] for p in ps]); hp = {}
            for rank, i in enumerate(order):
                hp[i] = min(1.0, ps[i][5] * (len(ps) - rank))
            for i, (x, y, dm, k, n, p) in enumerate(ps):
                print(f"     {x} - {y}: chenh {dm:+.3f}, thap hon {k}/{n}, p={p:.4f} (Holm {hp[i]:.4f})")
        lm = cur[(cur.rate == r) & (cur.cls == "All")].groupby("aqm")["lam_mean"].mean().round(2).to_dict()
        print("  lambda trung binh luc chay:", lm)
    # --- theo lop: nhan FRCNC co tao khac biet so voi doi chung KHONG nhan o cung diem van hanh khong? (chi tieu dinh truoc)
    print(f"\nTheo lop: pref_fix80 va pref_ctrl2 so voi {a.extra_name} (dinh truoc: Mice tre, Elephant mat goi, Elephant thong luong)")
    for r in sorted(res.rate.unique(), reverse=True):
        for cls in ("Mice", "Sheep", "Elephant"):
            row = []
            for m in ("delay_ms", "loss", "thr_mbps"):
                w = res[(res.rate == r) & (res.cls == cls)].pivot(index="seed", columns="aqm", values=m)
                row.append(" ".join(f"{k}={w[k].mean():.3f}" for k in ("pref_fix80", "pref_ctrl2", a.extra_name, "dn_dcod", "dcodel") if k in w))
            print(f"  {r:g} Mbps {cls:8s} tre: {row[0]} | mat goi: {row[1]} | thr: {row[2]}")
        for cls, m in (("Mice", "delay_ms"), ("Elephant", "loss"), ("Elephant", "thr_mbps")):
            w = res[(res.rate == r) & (res.cls == cls)].pivot(index="seed", columns="aqm", values=m)
            for x in ("pref_fix80", "pref_ctrl2"):
                for ref in (a.extra_name, "dn_dcod", "dcodel"):
                    if x in w and ref in w:
                        d = (w[x] - w[ref]).dropna()
                        if len(d) >= 5 and (d != 0).any():
                            print(f"     {r:g} Mbps {cls} {m}: {x} - {ref} = {d.mean():+.4f}, thap hon {int((d<0).sum())}/{len(d)}, p={wilcoxon(d).pvalue:.4f} (chua hieu chinh)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["ns3", "ukm"], default="ns3")
    ap.add_argument("--stage", choices=["train", "run", "report"], required=True)
    ap.add_argument("--label_mode", choices=["none", "external"], default="external")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--rates", type=float, nargs="+", default=[10.0, 8.0, 6.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--ukm_rate", type=float, default=2.0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tmp_dir", default="/content/_tmp_pref")
    ap.add_argument("--base_csv", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shield_ms", type=float, default=None, help="boc them bo bao ve (ep DROP khi sojourn > nguong), khong huan luyen lai")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--variant", default="", choices=["", "mice", "ukm2"], help="mice: beta Mice=(1.2,1.2,0.6); ket qua o results/pref_mice")
    ap.add_argument("--extra_csv", default=None, help="vd. results/dn_dcod/da_tai_shield10.csv (keep_shield nguong 10 ms)")
    ap.add_argument("--extra_name", default="keep_shield10")
    ap.add_argument("--aqms", nargs="+", default=None, choices=AQMS, help="chi chay cac co che nay (mac dinh: tat ca)")
    ap.add_argument("--model_dir", default=None, help="nap mo hinh tu thu muc khac (vd results/pref2 cua NS-3 de thu chuyen sang UKM, khong huan luyen)")
    ap.add_argument("--d_ref_ms", type=float, default=None, help="nguong tre bo dieu khien ctrl/ctrl2 (ms); mac dinh 0.1*D_MAX (NS-3 20 ms, UKM 100 ms)")
    ap.add_argument("--delay_pow", type=float, default=0.5)
    a = ap.parse_args()
    DELAY_POW = a.delay_pow
    if a.base_csv is None:
        a.base_csv = (os.path.join(DATA_DIR, "results", "dn_dcod", "da_tai_metrics.csv") if a.data == "ns3"
                      else os.path.join(DATA_DIR, "ukm_lop.csv"))
    {"train": train, "run": run, "report": report}[a.stage](a)
