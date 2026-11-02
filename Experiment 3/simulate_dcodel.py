"""
Mo phong dCoDel OFFLINE bang Python, replay tren vet goi tin DropTail
(out_ip_header_droptail_seed{S}_run{R}.csv) lam "nen trung lap" - dung
Y HET quyet dinh da ghi trong simulation_la50_v4.cc: dCoDel va DN-dCoD
KHONG cai dat thanh QueueDisc that trong NS-3, ma xu ly offline tren
Python, tai su dung vet DropTail.

Cai dat theo dung "Thuat toan 3: Thuat toan de xuat dCoDel" (Chuong 3,
cong thuc 3.50-3.52):
  - Target Delay dong (3.50): T_target -= alpha * (avg_queue_delay - T_target)
  - Interval dong (3.51):      T_interval = T_processed_last_interval / max(qlen,1)
  - Drop Time (3.52):          T_drop_next += T_interval / sqrt(max(qlen,1))
  - Dieu kien tha goi: sojourn_time > T_target VA now >= T_drop_next

GIOI HAN QUAN TRONG (can neu ro trong luan van khi trinh bay ket qua nay):
  - Day la MO PHONG XAP XI o muc hang doi, KHONG phai chay lai NS-3 that.
    Vong phan hoi TCP thuc (khi dCoDel tha goi khac DropTail, sender le ra
    se gui lai/dieu chinh cua so khac di) KHONG duoc mo phong - goi tin
    "den hang doi" van dung y het thoi diem cua DropTail.
  - payload_size khong duoc ghi truc tiep tai su kien ENQUEUE trong file
    goc, nen phai ghep noi (merge_asof, theo cap src_ip/dst_ip, gan nhat
    ve thoi gian) voi su kien TX de lay kich thuoc goi - co the sai lech
    nho khi 2 luong dung chung 1 cap IP hoat dong dong thoi.
  - T_max cho Target Delay: DA XAC DINH = 20ms (xem giai thich chi tiet o
    hang so T_MAX_TARGET_MS ben duoi) - gan voi nguong QoS cua lop Sheep/
    Elephant trong Bang 3.1, tranh hien tuong T_target "bao hoa tran" (da
    phat hien o phien ban truoc dung T_max=50ms: T_target nam o >=45ms toi
    66% thoi gian, khien dCoDel gan nhu khong bao gio tha goi).

Usage:
  python3 simulate_dcodel.py --seed 10 --run 1
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("LUANAN_DATA_DIR", "/content/drive/MyDrive/Colab/#sualuanan/data1")
BOTTLENECK_RATE_MBPS = 10.0     # tu simulation_la50_v4.cc: bottleneck.SetDeviceAttribute("DataRate","10Mbps")
QUEUE_MAX_PACKETS = 300          # tu simulation_la50_v4.cc: MaxSize QueueSize("300p")

# ---- Tham so dCoDel (cong thuc 3.50-3.52) ----
T_TARGET_INIT_MS = 5.0
T_INTERVAL_INIT_MS = 100.0
ALPHA = 0.15                     # he so dieu chinh Target Delay, luan van goi y 0.1-0.2
T_MIN_MS = 5.0                   # tu cong thuc (3.52): T_min = 5ms
T_MAX_TARGET_MS = 20.0           # DA XAC DINH (khong con la gia dinh) - xem giai thich duoi day.
                                  # Luan van khong cho so cu the cho T_max, chi noi "dung chung
                                  # cap nguong voi T_drop_next" (T_min=5ms). Phien ban truoc dung
                                  # T_max=50ms khien T_target "bao hoa tran" (>=45ms) toi 66% thoi
                                  # gian mo phong - vi cong thuc (3.50) lam T_target TANG moi khi
                                  # hang doi rang/do tre thap (rat thuong xay ra), nen chi mat vai
                                  # chuc ms de cham tran roi nam im o do, khien nguong tha goi luon
                                  # cao hon han do tre thuc te -> dCoDel gan nhu khong bao gio tha
                                  # goi, mau thuan voi ky vong cua Gia thuyet 4 ("dCoDel van phai
                                  # tha goi khi can de kiem soat that co chai"). Chon T_max=20ms vi
                                  # day la nguong gan voi yeu cau QoS cua lop Sheep/Elephant (Bang
                                  # 3.1: do nhay tre trung binh/thap) - du rong hon nhieu so voi
                                  # T_min=5ms (danh cho Mice, do nhay tre rat cao) de dCoDel van co
                                  # khong gian noi long khi gap luong Elephant, nhung khong rong toi
                                  # muc co che tha goi mat tac dung nhu voi 50ms.
T_INTERVAL_MAX_MS = 2 * T_INTERVAL_INIT_MS   # "khong vuot qua 2 lan interval mac dinh (100ms)"
TICK_MS = 1.0                    # "moi thoi gian (e.g., every 1ms)" theo Thuat toan 3

USECOLS = ["Timestamp", "event_type", "direction", "src_ip", "dst_ip",
           "protocol", "tos", "ttl", "identification", "fragment_offset",
           "MF", "DF", "payload_size", "tcp_ack_flag", "tcp_seq",
           "src_port", "dst_port"]


def load_arrivals(droptail_csv):
    """Doc vet DropTail, tra ve DataFrame cac goi tin DEN hang doi (ENQUEUE)
    kem payload_size (ghep noi tu su kien TX gan nhat cung cap src_ip/dst_ip),
    cho ca 2 huong (direction 0 va 1)."""
    print(f"[data] reading {droptail_csv} ...")
    df = pd.read_csv(droptail_csv, usecols=USECOLS, low_memory=False)

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
        print(f"[warn] {n_missing}/{len(merged)} goi khong ghep duoc payload_size "
              f"(ngoai tolerance 20ms) -> dien gia tri trung vi")
        merged["payload_size"] = merged["payload_size"].fillna(merged["payload_size"].median())
    merged["src_port"] = merged["src_port"].fillna(-1)
    merged["dst_port"] = merged["dst_port"].fillna(-1)
    print(f"[data] {len(merged)} goi ENQUEUE (ca 2 huong), "
          f"direction counts: {merged['direction'].value_counts().to_dict()}")
    return merged


def simulate_dcodel_one_direction(arrivals_dir, direction):
    """Mo phong 1 hang doi dCoDel FIFO don server, theo dung Thuat toan 3.
    Dung 1 doi thoi gian hop nhat (goi den + tick 1ms) de tranh vong lap
    long nhau de loi. arrivals_dir: DataFrame da sap xep theo Timestamp,
    can cot Timestamp (s), src_ip, dst_ip, src_port, dst_port, protocol,
    payload_size (bytes), tcp_ack_flag, tcp_seq.
    Tra ve list cac dict su kien (ENQUEUE/DEQUEUE/DROP/RX) dung schema chung."""
    rate_bps = BOTTLENECK_RATE_MBPS * 1e6

    state = dict(T_target_ms=T_TARGET_INIT_MS, T_interval_ms=T_INTERVAL_INIT_MS,
                 T_drop_next_ms=0.0, processed_time_acc_ms=0.0)
    queue = []          # FIFO: moi phan tu la dict co enqueue_ms + thong tin goi
    server_free_at_ms = [0.0]
    events = []

    arrivals = arrivals_dir.reset_index(drop=True)
    arrival_times_ms = arrivals["Timestamp"].to_numpy() * 1000.0
    n = len(arrivals)
    if n == 0:
        return events
    max_time_ms = arrival_times_ms[-1] + 2 * T_INTERVAL_INIT_MS  # du thoi gian xu ly not hang doi

    def serve_up_to(now_ms):
        """Cho server 'chay' den now_ms: goi nao da hoan tat truyen thi roi khoi hang doi."""
        while queue and server_free_at_ms[0] <= now_ms:
            p = queue[0]
            depart_ms = max(server_free_at_ms[0], p["enqueue_ms"])
            if depart_ms > now_ms:
                break
            xmit_ms = (p["payload_size"] * 8 / rate_bps) * 1000.0
            server_free_at_ms[0] = depart_ms + xmit_ms
            sojourn_ms = depart_ms - p["enqueue_ms"]
            state["processed_time_acc_ms"] += xmit_ms
            queue.pop(0)
            events.append(dict(Timestamp=depart_ms / 1000.0, event_type="DEQUEUE", direction=direction,
                                src_ip=p["src_ip"], dst_ip=p["dst_ip"], protocol=p["protocol"],
                                identification=0, payload_size=p["payload_size"],
                                tcp_ack_flag=p["tcp_ack_flag"], tcp_seq=p["tcp_seq"],
                                src_port=p["src_port"], dst_port=p["dst_port"],
                                queue_size=len(queue), d_queue_ms=sojourn_ms, rtt_ms=0))
            events.append(dict(Timestamp=depart_ms / 1000.0, event_type="RX", direction="",
                                src_ip=p["src_ip"], dst_ip=p["dst_ip"], protocol=p["protocol"],
                                identification=0, payload_size=p["payload_size"],
                                tcp_ack_flag=p["tcp_ack_flag"], tcp_seq=p["tcp_seq"],
                                src_port=p["src_port"], dst_port=p["dst_port"],
                                queue_size="", d_queue_ms="", rtt_ms=0))

    def do_tick(now_ms):
        """1 buoc Thuat toan 3: cap nhat T_target/T_interval/T_drop_next roi
        duyet hang doi de tha goi vuot nguong."""
        current_qlen = len(queue)
        avg_delay_ms = float(np.mean([now_ms - p["enqueue_ms"] for p in queue])) if current_qlen else 0.0

        state["T_target_ms"] = np.clip(
            state["T_target_ms"] - ALPHA * (avg_delay_ms - state["T_target_ms"]),
            T_MIN_MS, T_MAX_TARGET_MS)

        if current_qlen > 0:
            state["T_interval_ms"] = state["processed_time_acc_ms"] / max(current_qlen, 1)
        else:
            state["T_interval_ms"] = T_INTERVAL_INIT_MS
        state["T_interval_ms"] = float(np.clip(state["T_interval_ms"], TICK_MS, T_INTERVAL_MAX_MS))
        state["processed_time_acc_ms"] = 0.0

        if current_qlen > 0:
            state["T_drop_next_ms"] = float(np.clip(
                state["T_drop_next_ms"] + state["T_interval_ms"] / np.sqrt(max(current_qlen, 1)),
                T_MIN_MS, T_MAX_TARGET_MS))

        kept = []
        for p in queue:
            delay_ms = now_ms - p["enqueue_ms"]
            if delay_ms > state["T_target_ms"] and now_ms >= state["T_drop_next_ms"]:
                events.append(dict(Timestamp=now_ms / 1000.0, event_type="DROP", direction=direction,
                                    src_ip=p["src_ip"], dst_ip=p["dst_ip"], protocol=p["protocol"],
                                    identification=0, payload_size=p["payload_size"],
                                    tcp_ack_flag=p["tcp_ack_flag"], tcp_seq=p["tcp_seq"],
                                    src_port=p["src_port"], dst_port=p["dst_port"],
                                    queue_size=len(kept), d_queue_ms="", rtt_ms=0))
            else:
                kept.append(p)
        queue[:] = kept

    next_tick_ms = TICK_MS
    i = 0
    while i < n:
        arr_ms = arrival_times_ms[i]
        # chay het cac tick nam truoc thoi diem goi ke tiep den
        while next_tick_ms < arr_ms:
            serve_up_to(next_tick_ms)
            do_tick(next_tick_ms)
            next_tick_ms += TICK_MS
        serve_up_to(arr_ms)

        row = arrivals.iloc[i]
        # QUAN TRONG: phat 1 dong TX (payload_size day du) NGAY thoi diem
        # goi den, truoc ca nhanh ENQUEUE/DROP - de compute_metrics_one_run()
        # trong analyze_aqm_comparison.py ghep noi duoc payload_size cho
        # DEQUEUE (no chi doc payload_size tu TX gan nhat, khong doc truc
        # tiep tu DEQUEUE dau ra o day) - thieu dong nay se lam Thong luong
        # va Jain's Index tinh ra bang 0 het, du du lieu hang doi dung.
        events.append(dict(Timestamp=arr_ms / 1000.0, event_type="TX", direction="",
                            src_ip=row["src_ip"], dst_ip=row["dst_ip"], protocol=row["protocol"],
                            identification=0, payload_size=row["payload_size"],
                            tcp_ack_flag=row["tcp_ack_flag"], tcp_seq=row["tcp_seq"],
                            src_port=row["src_port"], dst_port=row["dst_port"],
                            queue_size="", d_queue_ms="", rtt_ms=0))
        if len(queue) < QUEUE_MAX_PACKETS:
            queue.append(dict(enqueue_ms=arr_ms, src_ip=row["src_ip"], dst_ip=row["dst_ip"],
                               protocol=row["protocol"], payload_size=row["payload_size"],
                               tcp_ack_flag=row["tcp_ack_flag"], tcp_seq=row["tcp_seq"],
                               src_port=row["src_port"], dst_port=row["dst_port"]))
            events.append(dict(Timestamp=arr_ms / 1000.0, event_type="ENQUEUE", direction=direction,
                                src_ip=row["src_ip"], dst_ip=row["dst_ip"], protocol=row["protocol"],
                                identification=0, payload_size=row["payload_size"],
                                tcp_ack_flag=row["tcp_ack_flag"], tcp_seq=row["tcp_seq"],
                                src_port=row["src_port"], dst_port=row["dst_port"],
                                queue_size=len(queue), d_queue_ms="", rtt_ms=0))
        else:
            events.append(dict(Timestamp=arr_ms / 1000.0, event_type="DROP", direction=direction,
                                src_ip=row["src_ip"], dst_ip=row["dst_ip"], protocol=row["protocol"],
                                identification=0, payload_size=row["payload_size"],
                                tcp_ack_flag=row["tcp_ack_flag"], tcp_seq=row["tcp_seq"],
                                src_port=row["src_port"], dst_port=row["dst_port"],
                                queue_size=len(queue), d_queue_ms="", rtt_ms=0))
        i += 1

    # xu ly not phan hang doi con lai sau goi cuoi cung
    while queue and next_tick_ms <= max_time_ms:
        serve_up_to(next_tick_ms)
        do_tick(next_tick_ms)
        next_tick_ms += TICK_MS
    serve_up_to(max_time_ms)

    return events


def run_dcodel_simulation(droptail_csv, out_csv):
    arrivals = load_arrivals(droptail_csv)
    all_events = []
    for direction in sorted(arrivals["direction"].dropna().unique()):
        sub = arrivals[arrivals["direction"] == direction].sort_values("Timestamp")
        print(f"[sim] direction={direction}: {len(sub)} goi den hang doi, dang mo phong dCoDel ...")
        evs = simulate_dcodel_one_direction(sub, direction)
        all_events.extend(evs)
        n_drop = sum(1 for e in evs if e["event_type"] == "DROP")
        n_deq = sum(1 for e in evs if e["event_type"] == "DEQUEUE")
        print(f"    -> {n_deq} DEQUEUE, {n_drop} DROP")

    out_df = pd.DataFrame(all_events)
    out_df.sort_values("Timestamp", inplace=True)
    cols = ["Timestamp", "event_type", "direction", "src_ip", "dst_ip", "protocol", "tos", "ttl",
            "identification", "fragment_offset", "MF", "DF", "payload_size", "tcp_ack_flag", "tcp_seq",
            "src_port", "dst_port", "queue_size", "d_queue_ms", "rtt_context", "rtt_ms"]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = ""
    out_df = out_df[cols]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\n[sim] saved -> {out_csv} ({len(out_df)} dong)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run", type=int, default=1)
    cli_args, _unknown = ap.parse_known_args()

    droptail_csv = f"{DATA_DIR}/out_ip_header_droptail_seed{cli_args.seed}_run{cli_args.run}.csv"
    out_csv = f"{DATA_DIR}/out_ip_header_dcodel_seed{cli_args.seed}_run{cli_args.run}.csv"
    run_dcodel_simulation(droptail_csv, out_csv)
