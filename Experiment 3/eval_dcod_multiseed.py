"""
Thuc nghiem 4 / Thuc nghiem 3 nhieu seed: danh gia dCoDel, DN-dCoD (khong nhan),
DN-dCoD (nhung nhan FRCNC) tren 10 vet DropTail (seed 1..10).

Quy trinh (cap nhat - TAI SU DUNG chinh sach da huan luyen):
  - Dung CHINH HAI CHINH SACH DN-dCoD da huan luyen tren vet seed 10 (job `dn_dcod_khong_nhan`
    va `dn_dcod`, file results/dn_dcod/{job}_qnet.pt) - cung la chinh sach da bao cao o Thuc
    nghiem 3, nen khong phai huan luyen lai.
  - Vi vet seed 10 la vet HUAN LUYEN, 10 seed danh gia la cac seed KHAC: 1..9 va 11
    (mac dinh cua --seeds). Seed 10 chi dung cho bang Thuc nghiem 3 (mot kich ban);
    script TU CHOI chay seed 10 tru khi them --allow_train_seed.
  - Voi MOI seed S trong --seeds:
        (1) dCoDel offline tren vet DropTail seed S      -> out_ip_header_dcodel_seed{S}_run{R}.csv
        (2) nhan FRCNC cho vet seed S                    -> frcnc_labels_droptail_seed{S}_run{R}.csv
        (3) chay GREEDY chinh sach da huan luyen (khong huan luyen them):
              dn_dcod_khong_nhan, dn_dcod                -> out_ip_header_{job}_seed{S}_run{R}.csv
    Buoc da co ket qua se duoc BO QUA (chay lai duoc sau khi Colab ngat).
  - Sau do:  python3 stat_tests_exp4.py --collect_aqm --seeds 1 2 3 4 5 6 7 8 9 11
             python3 stat_tests_exp4.py --wide_csv aqm_wide_avg_delay_ms.csv --lower_better ...

Yeu cau: vet NS-3 out_ip_header_{droptail,codel,fqcodel,pie,cake,red,ared}_seed{S}_run1.csv
(tao bang simulation_la50_v5_fixed.cc: --seed=S --run=1 --qdisc=..., cho S = 1..9 va 11), cung dcod_env.py,
train_dqn_dcod.py, run_dn_dcod_labeled.py, compute_frcnc_labels.py, simulate_dcodel.py,
analyze_aqm_comparison.py trong DATA_DIR.

Han che can neu trong luan an: ket luan chi ung voi MOT chinh sach da huan luyen (phuong sai do
huan luyen DQN khong duoc do); thay doi giua cac seed phan anh kha nang tong quat hoa sang kich ban
luu luong moi.
"""
import os
import sys
import gc
import random
from pathlib import Path

import numpy as np

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)
import importlib
importlib.invalidate_caches()

TRACE_DIR = os.environ.get("LUANAN_TRACE_DIR", DATA_DIR)   # vet NS-3 + ket qua dCoDel/DN-dCoD/nhan
RESULTS_DQN = f"{DATA_DIR}/results/dn_dcod"
TRAIN_SEED = 10
DEFAULT_FRCNC = f"{DATA_DIR}/results/ns3-6/frcnc/frcnc_ns3_agents50_seed0_best.pt"


def trace(aqm, seed, run):
    return f"{TRACE_DIR}/out_ip_header_{aqm}_seed{seed}_run{run}.csv"


def labels_path(seed, run):
    return f"{TRACE_DIR}/frcnc_labels_droptail_seed{seed}_run{run}.csv"


def load_policy(job, label_mode):
    import torch
    import train_dqn_dcod as tdd
    qpath = Path(RESULTS_DQN) / f"{job}_qnet.pt"
    if not qpath.exists():
        raise FileNotFoundError(f"Chua co {qpath} - huan luyen chinh sach truoc (xem docstring).")
    ck = torch.load(qpath, map_location=tdd.DEVICE)
    agent = tdd.DQNAgent(ck["state_dim"], ck["action_dim"])
    agent.q_net.load_state_dict(ck["model_state"])
    agent.q_net.eval()
    want = 4 if label_mode == "none" else 7
    assert ck["state_dim"] == want, f"{job}: state_dim={ck['state_dim']} khong khop label_mode={label_mode}"
    return agent


def main(seeds, run, job_none, job_label, frcnc_ckpt, skip_dcodel=False, allow_train_seed=False):
    import simulate_dcodel as sd
    from dcod_env import DCoDEnv
    import run_dn_dcod_labeled as rl
    import compute_frcnc_labels as cfl

    random.seed(0)
    np.random.seed(0)

    agent_none = load_policy(job_none, "none")
    agent_label = load_policy(job_label, "external")

    frcnc_model = scaler_mean = scaler_scale = None   # nap muc luoi, chi khi can tao nhan

    for s in seeds:
        if s == TRAIN_SEED and not allow_train_seed:
            print(f"[seed {s}] LA VET HUAN LUYEN -> bo qua (them --allow_train_seed neu chu y)")
            continue
        dt = trace("droptail", s, run)
        if not Path(dt).exists():
            print(f"[seed {s}] THIEU {dt} -> bo qua")
            continue
        print(f"\n===== seed {s} =====")

        # (1) dCoDel
        out = trace("dcodel", s, run)
        if skip_dcodel or Path(out).exists():
            print(f"[seed {s}] dCoDel: da co/bo qua")
        else:
            sd.run_dcodel_simulation(dt, out)

        # (2) nhan FRCNC
        lp = labels_path(s, run)
        if not Path(lp).exists():
            if frcnc_model is None:
                frcnc_model = cfl.load_frcnc(frcnc_ckpt)
                scaler_mean, scaler_scale = cfl.load_scaler()
            X, keys = cfl.build_windows_from_tx(dt)
            preds = cfl.predict_labels(frcnc_model, X, scaler_mean, scaler_scale)
            import pandas as pd
            pd.DataFrame([dict(src_ip=k[0], dst_ip=k[1], src_port=k[2], dst_port=k[3],
                               window_start_s=k[4], label=cfl.LABEL_NAMES[p])
                          for k, p in zip(keys, preds)]).to_csv(lp, index=False)
            print(f"[seed {s}] nhan FRCNC -> {lp}")
            del X
        else:
            print(f"[seed {s}] nhan FRCNC: da co")

        # (3) greedy DN-dCoD (khong nhan)
        out_n = trace("dn_dcod_khong_nhan", s, run)
        if not Path(out_n).exists():
            env = DCoDEnv(dt, direction=0, label_mode="none", max_steps=None)
            summ = env.run_greedy_and_export(agent_none, out_n)
            print(f"[seed {s}] DN-dCoD (khong nhan): "
                  f"{ {k: round(v, 4) for k, v in summ.items() if k != 'by_label' and isinstance(v, float)} }")
            del env
        else:
            print(f"[seed {s}] DN-dCoD (khong nhan): da co")

        # (3b) greedy DN-dCoD (nhung nhan)
        out_l = trace("dn_dcod", s, run)
        if not Path(out_l).exists():
            env = DCoDEnv(dt, direction=0, label_mode="external", max_steps=None)
            fn = rl.build_label_provider(lp)
            env.set_label_provider(fn)
            summ = env.run_greedy_and_export(agent_label, out_l)
            tot = max(1, fn.n_hits[0] + fn.n_miss[0])
            print(f"[seed {s}] DN-dCoD (nhung nhan): "
                  f"{ {k: round(v, 4) for k, v in summ.items() if k != 'by_label' and isinstance(v, float)} } "
                  f"| nhan khop {100 * fn.n_hits[0] / tot:.1f}%")
            del env
        else:
            print(f"[seed {s}] DN-dCoD (nhung nhan): da co")
        gc.collect()

    print("\nXong. Buoc tiep: python3 stat_tests_exp4.py --collect_aqm --seeds "
          + " ".join(str(x) for x in seeds if x != TRAIN_SEED))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--job_none", default="dn_dcod_khong_nhan")
    ap.add_argument("--job_label", default="dn_dcod")
    ap.add_argument("--frcnc_ckpt", default=DEFAULT_FRCNC)
    ap.add_argument("--skip_dcodel", action="store_true")
    ap.add_argument("--allow_train_seed", action="store_true")
    ap.add_argument("--policy_dir", default=None, help="thu muc chua {job}_qnet.pt (vd results/dn_dcod_v2); mac dinh results/dn_dcod")
    a, _ = ap.parse_known_args()
    if a.policy_dir:
        RESULTS_DQN = a.policy_dir
    main(a.seeds, a.run, a.job_none, a.job_label, a.frcnc_ckpt, a.skip_dcodel, a.allow_train_seed)
