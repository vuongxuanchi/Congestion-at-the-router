"""[ctrl3 = bo TU DONG: lambda0=0,5, D_ref tu hieu chuan tu 5 cua so dau, lambda ket trong [0,4; 0,6]]
Khac phuc 2 han che cua quet ctrl2 tren UKM: (1) chi 2 seed, chon sau khi biet ket qua; (2) chua so voi doi chung.
Cach lam: CHOT TRUOC cau hinh tu quet (chay dev 2,3): cua so 500, buoc 0,2, D_ref 100 ms (= 0,1*D_max, mac dinh).
Kiem tren cac run con lai 4..10 (n = 7, KHONG dung lai run 2,3 da dung de chon), so cap voi:
  pref_fix50 (ukm2, lambda = 0,5), ctrl2 mac dinh (cua so 1000, buoc 0,1), chi bo bao ve 50 ms (keep_shield).
Colab (moi lenh 1 cell; truoc do %cd {D}):
  !python3 kiem_ctrl3.py --stage run                       # ~7 lan chay ctrl2, 1 cau hinh
  !python3 kiem_ctrl3.py --stage report --shield_csv <duong dan csv co keep_shield nguong 50 ms>
"""
import os, sys, argparse
import numpy as np, pandas as pd
D = os.environ.get("LUANAN_DATA_DIR", ".")
sys.path.insert(0, D); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WIN, STEP, DREF = 500, 0.05, None             # ctrl3: cua so 500; D_ref TU DONG; cac hang so khac o pref_ctrl.C3_*
TEST = [2, 3, 4, 5, 6, 7, 8, 9, 10]         # chay ca 2..10; BAO CAO CHINH tren 4..10 (--runs 4 5 6 7 8 9 10)
OD = os.path.join(D, "results_ukm", "pref_ukm2")
TAG = "_ctrl3"


def run(a):
    import pref_ctrl as pc
    pc.CTRL_STEP, pc.CTRL_W = STEP, WIN
    ns = argparse.Namespace(data="ukm", stage="run", label_mode="external", episodes=0, rates=[2.0], seeds=[], runs=a.runs,
                            ukm_rate=2.0, out_dir=OD, tmp_dir="/content/_tmp_pref", base_csv=None, cpu=False, shield_ms=None,
                            tag=TAG, variant="ukm2", extra_csv=None, extra_name="x", aqms=["pref_ctrl3"], model_dir=OD,
                            d_ref_ms=DREF, delay_pow=0.5)
    pc.run(ns)


def report(a):
    from scipy.stats import wilcoxon
    allr = pd.read_csv(os.path.join(OD, "pref_metrics.csv"))
    tun = pd.read_csv(os.path.join(OD, f"pref_metrics{TAG}.csv")); tun["aqm"] = "ctrl3_tu_dong"
    t2 = pd.read_csv(os.path.join(OD, "pref_metrics_ctrl2tuned.csv")); t2["aqm"] = "ctrl2_chinh"
    parts = [allr[allr.aqm.isin(["pref_fix50", "pref_ctrl2"])], t2, tun]
    if a.shield_csv and os.path.exists(a.shield_csv):
        sh = pd.read_csv(a.shield_csv); parts.append(sh[sh.aqm == "keep_shield"])
    else:
        import glob
        print("!! thieu --shield_csv -> bo so sanh voi bo bao ve. File ung vien (chon file nguong 50 ms):")
        for f in sorted(glob.glob(os.path.join(D, "results_ukm", "**", "*.csv"), recursive=True)):
            try:
                if "keep_shield" in set(pd.read_csv(f, usecols=["aqm"]).aqm): print("   ", f)
            except Exception: pass
    r = pd.concat(parts, ignore_index=True); r = r[(r.cls == "All") & r.seed.isin(a.runs)]
    print(f"Run kiem: {a.runs} (n toi da = {len(a.runs)})")
    W = {m: r.pivot_table(index="seed", columns="aqm", values=m) for m in ("delay_ms", "loss", "thr_mbps")}
    for m, w in W.items():
        print(f"\n{m}: " + " | ".join(f"{k} {(100*w[k] if m=='loss' else w[k]).mean():.3f} (n={w[k].count()})" for k in w))
    for ref in ("pref_fix50", "keep_shield", "ctrl2_chinh", "pref_ctrl2"):
        if ref not in W["delay_ms"]: continue
        print(f"\n--- ctrl3_tu_dong so voi {ref} (cap theo run) ---")
        ps = []
        for m in ("delay_ms", "loss", "thr_mbps"):
            d = (W[m]["ctrl3_tu_dong"] - W[m][ref]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            ps.append(p); print(f"  {m:9s}: chenh {d.mean():+.4f}, thap hon {int((d<0).sum())}/{len(d)}, p = {p:.4f}")
        ok = [p for p in ps if not np.isnan(p)]
        if ok: print(f"  Holm (3 chi tieu): p_nho_nhat x {len(ok)} = {min(1, min(ok)*len(ok)):.4f}")
        dl = (W["delay_ms"]["ctrl3_tu_dong"] - W["delay_ms"][ref]).dropna(); ls = (W["loss"]["ctrl3_tu_dong"] - W["loss"][ref]).dropna()
        both = int(((dl < 0) & (ls.reindex(dl.index) < 0)).sum())
        print(f"  tot hon CA tre VA mat goi: {both}/{len(dl)} run  (ap dao that su = {len(dl)}/{len(dl)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "report"], required=True)
    ap.add_argument("--runs", type=int, nargs="+", default=None)
    ap.add_argument("--shield_csv", nargs="?", default=None)
    a = ap.parse_args(); a.runs = a.runs or (TEST if a.stage == "run" else [4, 5, 6, 7, 8, 9, 10]); run(a) if a.stage == "run" else report(a)
