"""Quet 3 bien cua bo TU DONG ctrl3 -> tre va mat goi (khong huan luyen, chi kiem). D_ref khong con la bien (tu suy ra).
3 bien: cua so W (goi), buoc lambda (C3_STEP), he so nguong kappa (C3_K: D_ref = kappa x trung vi tre 5 cua so dau).
Chay (Colab, moi lenh 1 cell; truoc do: %cd {D}):
  !python3 sweep_ctrl3.py --data ukm        # 27 cau hinh x 2 run ~ 1,8 gio  (them --seeds 2 de con ~0,9 gio)
  !python3 sweep_ctrl3.py --data ns3        # 27 cau hinh x 2 seed ~ 3,9 gio (nen chay UKM truoc)
Ngat/chay lai duoc: cau hinh/seed da xong se bo qua. Dung seed phat trien 2,3 (khong dung cac seed kiem 4..10 / 4..9,11).
Ve: chay ve_ctrl3_sweep_3d.py (khong tham so).
"""
import os, sys, argparse
import pandas as pd
D = os.environ.get("LUANAN_DATA_DIR", ".")
sys.path.insert(0, D); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GRID = dict(win=[250, 500, 1000], step=[0.025, 0.05, 0.1], kap=[1.25, 1.5, 2.0])


def run(a):
    import pref_ctrl as pc
    csv = os.path.join(D, f"sweep_ctrl3_{a.data}.csv")
    res = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    ukm = a.data == "ukm"
    for kap in GRID["kap"]:
        for step in GRID["step"]:
            for win in GRID["win"]:
                od = os.path.join(D, "sweep_tmp3", a.data, f"k{kap:g}_s{step:g}_w{win}"); os.makedirs(od, exist_ok=True)
                pc.CTRL_W, pc.C3_STEP, pc.C3_K = win, step, kap
                ns = argparse.Namespace(data=a.data, stage="run", label_mode="external", episodes=0, rates=[10.0], seeds=a.seeds, runs=a.seeds,
                                        ukm_rate=2.0, out_dir=od, tmp_dir="/content/_tmp_pref", base_csv=None, cpu=False, shield_ms=None, tag=None,
                                        variant="ukm2" if ukm else "", extra_csv=None, extra_name="x", aqms=["pref_ctrl3"],
                                        model_dir=os.path.join(D, "results_ukm/pref_ukm2" if ukm else "results/pref2"), d_ref_ms=None, delay_pow=0.5)
                pc.run(ns)
                f = os.path.join(od, "pref_metrics.csv")
                if not os.path.exists(f): continue
                m = pd.read_csv(f); m = m[(m.cls == "All") & (m.aqm == "pref_ctrl3")].copy()
                m["kap"], m["step"], m["win"] = kap, step, win
                res = pd.concat([res, m], ignore_index=True).drop_duplicates(["kap", "step", "win", "seed"])
                res.to_csv(csv, index=False)
    print("Xong ->", csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["ns3", "ukm"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    run(ap.parse_args())
