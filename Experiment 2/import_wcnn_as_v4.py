"""
Nhap 10 log wcnn_seed*.json DA CO SAN (tu train_ns3_multiseed.py) vao dung
dinh dang ma summarize() trong train_wcnn_ablation.py can doc, gan nhan la
buoc "v4_multiwave" - vi kien truc/sieu tham so cua WCNN hien tai (Wav-KAN
3 thanh phan, tach LR + warm-up/cosine, EMA) khop CHINH XAC voi dinh nghia
v4_multiwave, khong can huan luyen lai.

Chi lam MOT viec: doc training_logs/wcnn_seed{s}.json, ghi lai thanh
training_logs/ablation_v4_multiwave_seed{s}.json (giu nguyen "history" va
"test_metrics" - day la 2 truong duy nhat summarize() dung toi).

Usage:
  python3 import_wcnn_as_v4.py
"""
import os
import json
from pathlib import Path

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
TRAINING_LOG_DIR = Path(f"{DATA_DIR}/results/ns3-6/training_logs")

N_SEEDS = 10
copied, missing = 0, []

for s in range(N_SEEDS):
    src = TRAINING_LOG_DIR / f"wcnn_seed{s}.json"
    dst = TRAINING_LOG_DIR / f"ablation_v4_multiwave_seed{s}.json"
    if not src.exists():
        missing.append(str(src))
        continue
    d = json.loads(src.read_text())
    out = {
        "job_name": f"ablation_v4_multiwave_seed{s}",
        "version": "v4_multiwave",
        "run_seed": s,
        "history": d["history"],
        "best_val_acc": d["best_val_acc"],
        "test_acc": d["test_acc"],
        "test_loss": d["test_loss"],
        "test_metrics": d["test_metrics"],
    }
    dst.write_text(json.dumps(out, indent=2))
    copied += 1
    print(f"  wcnn_seed{s}.json -> ablation_v4_multiwave_seed{s}.json  (test_acc={d['test_acc']:.4f})")

print(f"\nDa nhap {copied}/{N_SEEDS} seed cho v4_multiwave.")
if missing:
    print(f"Thieu {len(missing)} file:")
    for m in missing:
        print(f"  {m}")
else:
    print("Du 10 seed - chi can chay them v1_base, v2_lr_sched, v3_ema:")
    print("  !python train_wcnn_ablation.py --n_seeds 10 --epochs 50 "
          "--versions v1_base,v2_lr_sched,v3_ema")
    print("roi chay lai phan summarize()/plot_ablation() trong "
          "train_wcnn_ablation.py voi versions=\"v1_base,v2_lr_sched,v3_ema,v4_multiwave\" "
          "(khong can --n_seeds/--epochs vi da co du log).")
