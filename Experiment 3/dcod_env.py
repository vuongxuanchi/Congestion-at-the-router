"""
Environment kieu Gym cho DN-dCoD (co nhan / khong nhan) (Thuat toan DQN + dCoDel dong,
Chuong 3 muc 3.6.2), replay tren vet goi tin DropTail (dung lam nen trung
lap, giong het simulate_dcodel.py) - MOI goi tin DEN la 1 buoc (step).

Trang thai (cong thuc 3.47):
  s_t = {Label_FRCNC, Sojourn_Time, Q_length, Delta_Delay, Rate}
Script nay dung LABEL_MODE="none" (DN-dCoD khong nhan, bo Label_FRCNC ra khoi state -
day la phien ban dang chay o luot nay) hoac LABEL_MODE="external" (DN-dCoD,
nhan nhan tu 1 ham/gia tri ben ngoai truyen vao qua set_label_provider() -
se noi vao o buoc sau, dung ns3_window_6.csv + checkpoint FRCNC).

Hanh dong (4 hanh dong a0-a3, Chuong 3 muc 3.6.2):
  a0: Giu goi (Keep)               - khong can thiep, xu ly binh thuong
  a1: Loai bo goi (Drop)           - chu dong tha goi dang xet
  a2: Tang T_target                - noi long do tre (uu tien thong luong)
  a3: Giam T_target                - siet chat do tre (uu tien QoS thap tre)

Phan thuong (cong thuc 3.48), tinh tren cua so goi gan nhat (rolling window):
  R = -(beta1*D_avg/D_max + beta2*Loss_Rate + beta3*(1 - Throughput/C))
beta1/beta2/beta3 phu thuoc nhan (Mice/Sheep/Elephant) khi LABEL_MODE=
"external"; khi LABEL_MODE="none" (DN-dCoD khong nhan) dung bo trong so CAN BANG
(giong logic "Sheep" trong luan van) cho MOI goi, vi khong co nhan de
phan biet.

GIOI HAN (giong het simulate_dcodel.py, ke lai o day cho day du):
  - Mo phong hang doi xap xi, KHONG phai NS-3 that - khong co phan hoi TCP
    thuc khi hanh dong drop/giu khac DropTail.
  - payload_size ghep noi tu TX gan nhat cung cap src_ip/dst_ip (co the
    sai lech nho khi nhieu luong dung chung 1 cap IP).
"""
import os
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
BOTTLENECK_RATE_MBPS = 10.0
QUEUE_MAX_PACKETS = 300

T_TARGET_INIT_MS = 5.0
T_MIN_MS = 5.0
T_MAX_TARGET_MS = 50.0          # GIA DINH giong simulate_dcodel.py - can xac nhan lai
T_TARGET_STEP_MS = 1.0          # buoc tang/giam khi hanh dong a2/a3

D_MAX_MS = 200.0                # chuan hoa D_avg/D_max trong reward (3.48) - GIA DINH ky thuat
ROLLING_WINDOW = 50              # so goi gan nhat dung de tinh D_avg/Loss_Rate/Throughput cho reward

# beta mac dinh khi KHONG co nhan (DN-dCoD khong nhan) - can bang ca 3 muc tieu,
# giong dung logic "Sheep" da mo ta cho truong hop co nhan (Chuong 3).
BETA_NO_LABEL = dict(beta1=1.0, beta2=1.0, beta3=1.0)
# beta theo nhan (dung khi LABEL_MODE="external") - CHUAN HOA tong = 3.0 cho
# CA 3 lop (giong het BETA_NO_LABEL: 1+1+1=3), giu NGUYEN TY LE uu tien ban
# dau (Mice 3:1:1, Elephant 1:2:2) nhung doi do lon tuyet doi. Neu KHONG
# chuan hoa (ban truoc: Mice va Elephant co tong=5, Sheep tong=3), phan
# thuong bi lech thang do khac nhau tuy nhan - vi ~70% luu luong la Mice+
# Elephant (theo phan bo FRCNC thuc te), da so buoc bi phat NANG HON 67%
# so voi truong hop khong nhan, khien reward tong DN-dCoD (co nhan) nhin
# "te hon" DN-dCoD (khong nhan) chi vi thang do, khong han la chinh sach
# dieu khien thuc su kem hon. Rieng beta1=3.0 rat lon so voi beta2/beta3=1.0
# o Mice con tao dong luc "danh doi drop lay giam tre" (tha goi loai no
# khoi trung binh do tre ngay lap tuc) - chuan hoa giam bot muc chenh lech
# tuyet doi nay ma van giu dung tinh than "Mice uu tien do tre gap 3 lan".
BETA_BY_LABEL = {
    "Mice":     dict(beta1=1.8, beta2=0.6, beta3=0.6),   # ty le 3:1:1 (uu tien do tre), tong=3.0
    "Sheep":    dict(beta1=1.0, beta2=1.0, beta3=1.0),   # can bang, tong=3.0
    "Elephant": dict(beta1=0.6, beta2=1.2, beta3=1.2),   # ty le 1:2:2 (uu tien thong luong/it mat goi), tong=3.0
}

ACTIONS = ["KEEP", "DROP", "INCREASE_TARGET", "DECREASE_TARGET"]
LABEL_NAMES_ = ["Mice", "Sheep", "Elephant"]


def load_arrivals(droptail_csv):
    """Giong het simulate_dcodel.py::load_arrivals - doc vet DropTail, ghep
    payload_size tu TX gan nhat, tra ve DataFrame goi ENQUEUE ca 2 huong."""
    print(f"[env-data] reading {droptail_csv} ...")
    usecols = ["Timestamp", "event_type", "direction", "src_ip", "dst_ip",
               "protocol", "payload_size", "tcp_ack_flag", "tcp_seq", "src_port", "dst_port"]
    df = pd.read_csv(droptail_csv, usecols=usecols, low_memory=False)

    tx = df[df["event_type"] == "TX"][
        ["Timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
         "payload_size", "tcp_ack_flag", "tcp_seq"]
    ].sort_values("Timestamp")

    enq = df[df["event_type"] == "ENQUEUE"][
        ["Timestamp", "direction", "src_ip", "dst_ip"]
    ].sort_values("Timestamp").reset_index(drop=True)

    merged = pd.merge_asof(enq, tx, on="Timestamp", by=["src_ip", "dst_ip"],
                            direction="nearest", tolerance=0.02)
    n_missing = merged["payload_size"].isna().sum()
    if n_missing:
        merged["payload_size"] = merged["payload_size"].fillna(merged["payload_size"].median())
    merged["src_port"] = merged["src_port"].fillna(-1)
    merged["dst_port"] = merged["dst_port"].fillna(-1)
    print(f"[env-data] {len(merged)} goi ENQUEUE, direction counts: "
          f"{merged['direction'].value_counts().to_dict()}")
    return merged


class DCoDEnv:
    """1 episode = di het (hoac 1 doan cua) chuoi goi den cua 1 huong. Moi
    step la 1 goi tin: agent chon hanh dong cho goi DANG XET (dau hang doi
    neu co hang doi, hoac chinh goi vua den), moi truong cap nhat hang doi/
    server, tra ve trang thai ke tiep + phan thuong."""

    def __init__(self, droptail_csv, direction=0, label_mode="none",
                 max_steps=None, rolling_window=ROLLING_WINDOW):
        self.label_mode = label_mode          # "none" (DN-dCoD khong nhan) hoac "external" (DN-dCoD co nhan)
        self.label_provider = None            # ham(row) -> "Mice"/"Sheep"/"Elephant", gan sau
        self.rate_bps = BOTTLENECK_RATE_MBPS * 1e6

        arrivals = load_arrivals(droptail_csv)
        self.arrivals = arrivals[arrivals["direction"] == direction].sort_values(
            "Timestamp").reset_index(drop=True)
        if max_steps:
            self.arrivals = self.arrivals.iloc[:max_steps].reset_index(drop=True)
        self.n = len(self.arrivals)
        self.direction = direction

        self.state_dim = 4 if label_mode == "none" else 4 + 3   # +3 = one-hot Mice/Sheep/Elephant
        self.action_dim = len(ACTIONS)

    def set_label_provider(self, fn):
        """fn(row) -> "Mice"/"Sheep"/"Elephant" - dung o buoc gan nhan FRCNC
        sau nay (label_mode phai la "external")."""
        self.label_provider = fn

    def reset(self):
        self.i = 0
        self.T_target_ms = T_TARGET_INIT_MS
        self.queue = []                     # FIFO cac goi dang cho: dict co enqueue_ms, payload_size,...
        self.server_free_at_ms = 0.0
        self.recent_delays_ms = deque(maxlen=ROLLING_WINDOW)
        self.recent_drops = deque(maxlen=ROLLING_WINDOW)
        self.recent_bytes = deque(maxlen=ROLLING_WINDOW)
        self.recent_times_s = deque(maxlen=ROLLING_WINDOW)
        self.total_enqueue = 0
        self.total_drop = 0
        self.total_deliver_bytes = 0.0
        self.episode_start_s = float(self.arrivals["Timestamp"].iloc[0]) if self.n else 0.0
        self.episode_end_s = float(self.arrivals["Timestamp"].iloc[-1]) if self.n else 0.0
        # --- theo doi RIENG theo tung nhan (Mice/Sheep/Elephant) - de tra
        # loi cau hoi: mat goi tang them (khi co nhan) co TAP TRUNG dung
        # vao nhan duoc uu tien do tre (vd Mice) hay dan deu bat ke nhan -
        # phan biet "danh doi co chu dich" voi "tac dung phu khong mong
        # muon". Chi co y nghia khi label_mode="external".
        self.label_stats = {c: dict(n_enqueue=0, n_drop=0, n_deliver=0, total_delay_ms=0.0)
                             for c in LABEL_NAMES_}
        return self._current_state()

    def _get_current_label(self):
        """Nhan cua goi DANG XET (self.i) - dung chung cho _current_state(),
        _current_beta() va step() de KHONG goi label_provider() nhieu lan
        thua cho cung 1 goi (label_provider co the tinh toan/tra cuu ton
        kem)."""
        if self.label_mode != "external" or self.label_provider is None or self.i >= self.n:
            return "Sheep"
        try:
            return self.label_provider(self.arrivals.iloc[self.i])
        except Exception:
            return "Sheep"

    def _serve_up_to(self, now_ms):
        while self.queue and self.server_free_at_ms <= now_ms:
            p = self.queue[0]
            depart_ms = max(self.server_free_at_ms, p["enqueue_ms"])
            if depart_ms > now_ms:
                break
            xmit_ms = (p["payload_size"] * 8 / self.rate_bps) * 1000.0
            self.server_free_at_ms = depart_ms + xmit_ms
            sojourn_ms = depart_ms - p["enqueue_ms"]
            self.queue.pop(0)
            self.recent_delays_ms.append(sojourn_ms)
            self.recent_drops.append(0)
            self.recent_bytes.append(p["payload_size"])
            self.recent_times_s.append(depart_ms / 1000.0)
            self.total_deliver_bytes += p["payload_size"]
            if getattr(self, "_export_events", None) is not None:      # chi khi dang xuat vet
                _t = depart_ms / 1000.0
                self._export_events.append(dict(Timestamp=_t, event_type="DEQUEUE", direction=self.direction,
                    src_ip=p["src_ip"], dst_ip=p["dst_ip"], protocol="", identification=0,
                    payload_size=p["payload_size"], tcp_ack_flag="", tcp_seq="", src_port=p.get("src_port", ""), dst_port=p.get("dst_port", ""),
                    queue_size=len(self.queue), d_queue_ms=sojourn_ms, rtt_ms=0))
                self._export_events.append(dict(Timestamp=_t, event_type="RX", direction="",
                    src_ip=p["src_ip"], dst_ip=p["dst_ip"], protocol="", identification=0,
                    payload_size=p["payload_size"], tcp_ack_flag="", tcp_seq="", src_port=p.get("src_port", ""), dst_port=p.get("dst_port", ""),
                    queue_size="", d_queue_ms="", rtt_ms=0))
            lbl = p.get("label", "Sheep")
            if lbl in self.label_stats:
                self.label_stats[lbl]["n_deliver"] += 1
                self.label_stats[lbl]["total_delay_ms"] += sojourn_ms

    def _current_state(self):
        """4 dac trung vat ly (+ 3 one-hot nhan neu label_mode="external"):
        sojourn_time uoc tinh cho goi sap toi (dua tren backlog hien tai),
        do dai hang doi, chenh lech do tre so voi T_target, toc do goi
        den gan day (goi/s)."""
        qlen = len(self.queue)
        backlog_bits = sum(p["payload_size"] for p in self.queue) * 8
        est_sojourn_ms = (backlog_bits / self.rate_bps) * 1000.0 if qlen else 0.0
        delta_delay = est_sojourn_ms - self.T_target_ms
        if len(self.recent_times_s) >= 2:
            span = self.recent_times_s[-1] - self.recent_times_s[0]
            rate_pps = len(self.recent_times_s) / span if span > 1e-6 else 0.0
        else:
            rate_pps = 0.0

        state = [est_sojourn_ms / 100.0, qlen / QUEUE_MAX_PACKETS,
                 delta_delay / 100.0, rate_pps / 1000.0]

        if self.label_mode == "external":
            label = self._get_current_label()
            onehot = [1.0 if label == c else 0.0 for c in ["Mice", "Sheep", "Elephant"]]
            state = state + onehot
        return np.array(state, dtype=np.float32)

    def _current_beta(self):
        if self.label_mode == "external":
            label = self._get_current_label()
            return BETA_BY_LABEL.get(label, BETA_NO_LABEL)
        return BETA_NO_LABEL

    def _reward(self, beta):
        """Cong thuc (3.48): R = -(b1*D_avg/D_max + b2*Loss_Rate + b3*(1-Thr/C)).
        beta duoc TRUYEN VAO tu step() (da tinh san tu nhan cua GOI VUA
        XU LY) - KHONG tu goi _current_beta() o day nua, vi luc ham nay
        chay thi self.i DA TANG (tro sang goi tiep theo), goi lai se ra
        beta cua goi SAI (goi ke tiep, khong phai goi vua quyet dinh
        drop/giu)."""
        d_avg = float(np.mean(self.recent_delays_ms)) if self.recent_delays_ms else 0.0
        loss_rate = float(np.mean(self.recent_drops)) if self.recent_drops else 0.0
        if len(self.recent_times_s) >= 2:
            span = self.recent_times_s[-1] - self.recent_times_s[0]
            thr_mbps = (sum(self.recent_bytes) * 8 / 1e6) / span if span > 1e-6 else 0.0
        else:
            thr_mbps = 0.0
        penalty = (beta["beta1"] * min(d_avg / D_MAX_MS, 1.0)
                   + beta["beta2"] * loss_rate
                   + beta["beta3"] * (1.0 - min(thr_mbps / BOTTLENECK_RATE_MBPS, 1.0)))
        return -penalty

    def step(self, action):
        """action: 0=KEEP,1=DROP,2=INCREASE_TARGET,3=DECREASE_TARGET.
        Ap dung cho goi DANG DEN (row tai self.i); sau do cho server chay
        toi thoi diem goi nay, cap nhat hang doi, tra ve (next_state,
        reward, done, info)."""
        if self.i >= self.n:
            return self._current_state(), 0.0, True, {}

        row = self.arrivals.iloc[self.i]
        arr_ms = float(row["Timestamp"]) * 1000.0
        self._serve_up_to(arr_ms)

        current_label = self._get_current_label()   # 1 lan duy nhat cho goi nay, dung lai ben duoi
        current_beta = (BETA_BY_LABEL.get(current_label, BETA_NO_LABEL)
                         if self.label_mode == "external" else BETA_NO_LABEL)

        if action == 2:      # INCREASE_TARGET
            self.T_target_ms = float(np.clip(self.T_target_ms + T_TARGET_STEP_MS, T_MIN_MS, T_MAX_TARGET_MS))
        elif action == 3:    # DECREASE_TARGET
            self.T_target_ms = float(np.clip(self.T_target_ms - T_TARGET_STEP_MS, T_MIN_MS, T_MAX_TARGET_MS))

        do_drop = (action == 1)
        if not do_drop and len(self.queue) >= QUEUE_MAX_PACKETS:
            do_drop = True   # hang doi day -> buoc phai tha (tran bo dem)

        if getattr(self, "_export_events", None) is not None:           # chi khi dang xuat vet
            _t = arr_ms / 1000.0
            self._export_events.append(dict(Timestamp=_t, event_type="TX", direction="",
                src_ip=row["src_ip"], dst_ip=row["dst_ip"], protocol="", identification=0,
                payload_size=row["payload_size"], tcp_ack_flag="", tcp_seq="", src_port=row.get("src_port", ""), dst_port=row.get("dst_port", ""),
                queue_size="", d_queue_ms="", rtt_ms=0))
            self._export_events.append(dict(Timestamp=_t, event_type="DROP" if do_drop else "ENQUEUE",
                direction=self.direction, src_ip=row["src_ip"], dst_ip=row["dst_ip"], protocol="",
                identification=0, payload_size=row["payload_size"] if do_drop else "",
                tcp_ack_flag="", tcp_seq="", src_port=row.get("src_port", ""), dst_port=row.get("dst_port", ""),
                queue_size=len(self.queue), d_queue_ms="", rtt_ms=0))

        self.total_enqueue += 1
        if current_label in self.label_stats:
            self.label_stats[current_label]["n_enqueue"] += 1
        if do_drop:
            self.total_drop += 1
            if current_label in self.label_stats:
                self.label_stats[current_label]["n_drop"] += 1
            self.recent_delays_ms.append(0.0)
            self.recent_drops.append(1)
            self.recent_bytes.append(0.0)
            self.recent_times_s.append(arr_ms / 1000.0)
        else:
            self.queue.append(dict(enqueue_ms=arr_ms, payload_size=float(row["payload_size"]),
                                    src_ip=row["src_ip"], dst_ip=row["dst_ip"], label=current_label,
                                    src_port=row.get("src_port", -1), dst_port=row.get("dst_port", -1)))

        self.i += 1
        reward = self._reward(current_beta)
        done = self.i >= self.n
        next_state = self._current_state()
        info = dict(T_target_ms=self.T_target_ms, qlen=len(self.queue),
                    total_drop=self.total_drop, total_enqueue=self.total_enqueue)
        return next_state, reward, done, info

    def episode_summary(self):
        """Tom tat cuoi episode, cung dinh dang voi analyze_aqm_comparison.py
        de so sanh truc tiep trong Thuc nghiem 3. Them "by_label": phan
        tich rieng delay/loss theo tung nhan (chi co y nghia khi
        label_mode="external") - dung de kiem tra mat goi tang them (khi
        co nhan) co TAP TRUNG dung vao nhan duoc uu tien do tre hay khong."""
        avg_delay_ms = float(np.mean(self.recent_delays_ms)) if self.recent_delays_ms else float("nan")
        loss_rate = self.total_drop / max(1, self.total_enqueue)
        duration_s = max(1e-6, self.episode_end_s - self.episode_start_s)
        throughput_mbps = (self.total_deliver_bytes * 8 / 1e6) / duration_s

        by_label = {}
        for c, s in self.label_stats.items():
            n_enq = s["n_enqueue"]
            by_label[c] = dict(
                n_enqueue=n_enq, n_drop=s["n_drop"], n_deliver=s["n_deliver"],
                loss_rate=(s["n_drop"] / n_enq) if n_enq > 0 else float("nan"),
                avg_delay_ms=(s["total_delay_ms"] / s["n_deliver"]) if s["n_deliver"] > 0 else float("nan"),
                share_of_traffic=(n_enq / max(1, self.total_enqueue)),
            )

        return dict(avg_delay_ms=avg_delay_ms, loss_rate=loss_rate,
                    throughput_mbps=throughput_mbps, n_packets=self.total_enqueue,
                    n_drop=self.total_drop, final_T_target_ms=self.T_target_ms,
                    by_label=by_label)

    def run_greedy_and_export(self, agent, out_csv):
        """Chay 1 lan DUY NHAT qua toan bo du lieu (khong epsilon-explore,
        luon chon hanh dong tot nhat theo Q-network da huan luyen), ghi lai
        du lieu ENQUEUE/DEQUEUE/DROP DUNG SCHEMA voi out_ip_header_*.csv de
        cam thang vao analyze_aqm_comparison.py (Thuc nghiem 3)."""
        events = []
        s = self.reset()
        self._export_events = events      # DEQUEUE/RX duoc ghi ngay trong _serve_up_to (dung thoi diem that)
        while True:
            a = agent.select_action(s, greedy=True)
            s2, r, done, info = self.step(a)

            s = s2
            if done:
                break

        self._serve_up_to(float("inf"))   # xa not cac goi con trong hang doi o cuoi vet (dCoDel cung xa het)
        self._export_events = None
        out_df = pd.DataFrame(events)
        cols = ["Timestamp", "event_type", "direction", "src_ip", "dst_ip", "protocol", "tos", "ttl",
                "identification", "fragment_offset", "MF", "DF", "payload_size", "tcp_ack_flag", "tcp_seq",
                "src_port", "dst_port", "queue_size", "d_queue_ms", "rtt_context", "rtt_ms"]
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df = out_df[cols].sort_values("Timestamp", kind="mergesort")
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"[env] xuat {len(out_df)} dong -> {out_csv}")
        return self.episode_summary()
