"""ctrl3 (tu dong: lambda0=0,5; D_ref = 1,5 x trung vi tre 5 cua so dau; lambda ket trong [0,4; 0,6]) tren NS-3, 10 Mbps.
GIU NGUYEN hang so da chot tren UKM (khong chinh lai): cua so 500, buoc 0,05 (xem pref_ctrl.C3_*). Kiem tra "mot cau hinh dung chung cho 2 mang".
Seed chinh thuc 2..9, 11 (n = 9; seed 1 la seed phat trien, seed 10 la seed huan luyen -> loai).
Colab (moi lenh 1 cell; truoc do %cd {D}):
  !python3 kiem_ctrl3_ns3.py --stage run
  !python3 kiem_ctrl3_ns3.py --stage report                                  # so voi lambda co dinh 0,5 va ctrl2 mac dinh
  !python3 kiem_ctrl3_ns3.py --stage report --shield_csv <csv co keep_shield> # them doi chung bo bao ve (tuy chon)
"""
import os, sys, argparse
import numpy as np, pandas as pd
D = os.environ.get("LUANAN_DATA_DIR", ".")
sys.path.insert(0, D); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OD = os.path.join(D, "results", "pref2"); TAG = "_ctrl3"; RATE = 10.0
SEEDS = [2, 3, 4, 5, 6, 7, 8, 9, 11]


def run(a):
    import pref_ctrl as pc
    pc.CTRL_W = 500
    ns = argparse.Namespace(data="ns3", stage="run", label_mode="external", episodes=0, rates=[RATE], seeds=a.seeds, runs=[],
                            ukm_rate=2.0, out_dir=OD, tmp_dir="/content/_tmp_pref", base_csv=None, cpu=False, shield_ms=None,
                            tag=TAG, variant="", extra_csv=None, extra_name="x", aqms=["pref_ctrl3"], model_dir=OD,
                            d_ref_ms=None, delay_pow=0.5)
    pc.run(ns)


def report(a):
    from scipy.stats import wilcoxon
    allr = pd.read_csv(os.path.join(OD, "pref_metrics.csv"))
    allr = allr[allr.aqm.isin(["pref_fix50", "pref_ctrl2"]) & (allr.rate == RATE)]
    c3 = pd.read_csv(os.path.join(OD, f"pref_metrics{TAG}.csv")); c3 = c3[c3.rate == RATE].copy(); c3["aqm"] = "ctrl3_tu_dong"
    parts = [allr, c3]
    if a.shield_csv and os.path.exists(a.shield_csv):
        sh = pd.read_csv(a.shield_csv); sh = sh[sh.aqm == "keep_shield"]
        if "rate" in sh: sh = sh[sh.rate == RATE]
        parts.append(sh)
    else:
        import glob
        print("(khong co --shield_csv -> bo doi chung bo bao ve. File ung vien:)")
        for f in sorted(glob.glob(os.path.join(D, "results", "**", "*.csv"), recursive=True)):
            try:
                if "keep_shield" in set(pd.read_csv(f, usecols=["aqm"]).aqm): print("   ", f)
            except Exception: pass
    r = pd.concat(parts, ignore_index=True); r = r[(r.cls == "All") & r.seed.isin(a.seeds)]
    print(f"NS-3 {RATE:g} Mbps, seed: {a.seeds}")
    W = {m: r.pivot_table(index="seed", columns="aqm", values=m) for m in ("delay_ms", "loss", "thr_mbps")}
    for m, w in W.items():
        print(f"\n{m}: " + " | ".join(f"{k} {(100*w[k] if m=='loss' else w[k]).mean():.3f} (n={w[k].count()})" for k in w))
    for ref in ("pref_fix50", "pref_ctrl2", "keep_shield"):
        if ref not in W["delay_ms"]: continue
        print(f"\n--- ctrl3_tu_dong so voi {ref} (cap theo seed) ---")
        ps = []
        for m in ("delay_ms", "loss", "thr_mbps"):
            d = (W[m]["ctrl3_tu_dong"] - W[m][ref]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            ps.append(p); print(f"  {m:9s}: chenh {d.mean():+.4f}, thap hon {int((d<0).sum())}/{len(d)}, p = {p:.4f}")
        ok = [p for p in ps if not np.isnan(p)]
        if ok: print(f"  Holm (3 chi tieu): p_nho_nhat x {len(ok)} = {min(1, min(ok)*len(ok)):.4f}")
    lm = c3[c3.cls == "All"].groupby("aqm")["lam_mean"].mean().round(2).to_dict(); print("\nlambda TB luc chay:", lm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["run", "report"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--shield_csv", nargs="?", default=None)
    a = ap.parse_args(); run(a) if a.stage == "run" else report(a)
