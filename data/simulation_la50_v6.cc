/*
 * simulation_la50_v2.cc — BẢN XÂY DỰNG LẠI theo đúng kịch bản gốc (sơ đồ) +
 * yêu cầu mới nhất: "TCP và UDP đều qua kết nối thắt cổ chai".
 *
 * ═══════════════════════════════════════════════════════════════════════
 * DANH SÁCH THAY ĐỔI SO VỚI BẢN GỐC (và LÝ DO)
 * ═══════════════════════════════════════════════════════════════════════
 * 1) TOPOLOGY GIỮ NGUYÊN (đúng theo cả sơ đồ lẫn yêu cầu):
 *      - Nốt 0–9   (N1–N10, 1-based) → nốt 10 (N11), access 20Mbps/5ms
 *      - Nốt 10–11 (N11–N12)          bottleneck 10Mbps/50ms
 *      - Nốt 12–49 (N13–N50)          → nốt 11 (N12), access 20Mbps/5ms
 *
 * 2) TRAFFIC — SỬA LẠI ĐỂ KHỚP SƠ ĐỒ GỐC + BẮT BUỘC QUA BOTTLENECK:
 *    Bản cũ: TẤT CẢ 10 nốt (0–9) đều gửi TCP; UDP lại đặt nhầm ở nốt 12–19
 *    (cùng phía N12, không hề đi qua bottleneck vì chỉ cách N12 đúng 1 hop).
 *    Bản mới, đúng theo sơ đồ:
 *      - TCP send : nốt 0–5   (N1–N6)   → TCP receive: nốt 12–17  (N13–N18)
 *      - UDP send : nốt 6–9   (N7–N10)  → UDP receive: nốt 46–49  (N47–N50)
 *    Vì bên gửi luôn ở phía N11 và bên nhận luôn ở phía N12, MỌI gói tin
 *    (cả TCP lẫn UDP) đều BẮT BUỘC phải đi qua link bottleneck (10↔11).
 *
 * 3) HÀNG ĐỢI BOTTLENECK = 300 GÓI (theo đúng yêu cầu "rớt gói khi hàng đợi
 *    (300 gói) đầy") — trước đây không set, dùng mặc định NS-3 (thường
 *    100p), nay set tường minh 300p qua bottleneck.SetQueue(...).
 *
 * 4) TẮT AQM TỰ ĐỘNG (FqCoDel) CỦA NS-3 TRÊN CẢ 2 LOẠI LINK:
 *    Từ NS-3 ~3.30, Ipv4AddressHelper::Assign() tự cài queue-disc mặc định
 *    (FqCoDel cho NetDevice 1 hàng đợi) trừ khi gọi DisableFlowControl().
 *    Vì mục đích nghiên cứu là tự cài đặt lại CoDel/dCoDeL/DQN bằng Python
 *    trên dữ liệu trace, cần đảm bảo NS-3 chỉ dùng hàng đợi DropTail THUẦN
 *    tại bottleneck (baseline trung lập) — gọi DisableFlowControl() trên cả
 *    hai PointToPointHelper trước khi Install().
 *
 * 5) GIỚI HẠN CỬA SỔ TCP — ĐÃ HIỆU CHỈNH: 1000 → 8192 → 16384 BYTE (16KB):
 *    Test với 1000 byte: TCP gần như đứng hình (0.16% traffic, gap tối đa
 *    ~16s — dấu hiệu vòng lặp timeout/retransmit 1s→2s→4s→8s→16s). Nguyên
 *    nhân: UDP 4×5Mbps=20Mbps đã tự vượt bottleneck 10Mbps 2 lần, TCP cửa
 *    sổ nhỏ (~66kbps/luồng lý thuyết) không đủ sức cạnh tranh.
 *    Chốt 16384 byte: RTT≈120ms → mỗi luồng TCP có tiềm năng ~1.09 Mbps
 *    (16384×8/0.12s), 6 luồng ≈ 6.5 Mbps tổng tiềm năng (~65% bottleneck)
 *    — đủ sức tranh chấp hàng đợi thật sự với UDP, vẫn là cửa sổ giới hạn
 *    rõ ràng so với mặc định NS-3 (~128KB).
 *
 * 6) TraceIpHeader GIỮ NGUYÊN (vẫn bắt Tx tại mọi nốt) — vì giờ traffic đi
 *    it nhất 2 hop (access → bottleneck → access), file CSV vẫn sẽ có
 *    nhiều dòng cho mỗi gói (theo từng hop), y hệt cơ chế trước — nhưng
 *    giờ CẢ TCP LẪN UDP đều có dòng TTL giảm dần qua bottleneck, nên khi xử
 *    lý bằng Python cần LỌC LẠI (bỏ điều kiện chỉ-nhận-protocol==6 cũ, vì
 *    giờ UDP cũng hợp lệ đi qua bottleneck).
 * 7) SONG CÔNG (DUPLEX) + GIẢM UDP — THÊM LẠI THEO YÊU CẦU MỚI NHẤT:
 *    Test song công với UDP=5Mbps/luồng cho thấy mỗi chiều bottleneck bị
 *    quá tải gấp 2 lần (20.19 Mbps / 10 Mbps), TCP chỉ chiếm 0.15% traffic
 *    dù đã nới cửa sổ lên 16KB — vì gốc rễ là UDP không co giãn, chiếm gần
 *    hết hàng đợi DropTail bất kể cửa sổ TCP bao nhiêu.
 *    => Giảm UDP: 5Mbps -> 2.2Mbps/luồng. 4 luồng x 2.2 = 8.8 Mbps/chiều,
 *    dưới bottleneck 10Mbps/chiều, để lại ~1.2Mbps/chiều cho TCP thật sự
 *    có cơ hội hoạt động. Giữ nguyên song công (cổng 9010/9011 cho chiều
 *    ngược) và cửa sổ TCP 16KB.
 *
 * 8) SEED — THÊM MỚI (yêu cầu mới nhất: "seed là 5"):
 *    Bản trước KHÔNG hề gọi RngSeedManager, nên NS-3 luôn dùng seed=1 mặc
 *    định (mọi lần chạy lại ra kết quả giống hệt nhau, không phản ánh tính
 *    ngẫu nhiên của traffic model OnOff/BulkSend). Nay gọi tường minh
 *    RngSeedManager::SetSeed(5) NGAY ĐẦU main(), trước khi tạo bất kỳ
 *    node/app nào (bắt buộc, vì RNG được cấp phát ngay khi đối tượng NS-3
 *    đầu tiên dùng đến nó — set seed trễ sẽ không có tác dụng với những gì
 *    đã khởi tạo trước đó). RngRun giữ mặc định = 1 (có thể đổi qua CLI
 *    "--run=N" nếu sau này cần chạy lặp lại nhiều lần với cùng seed=5 để
 *    lấy các mẫu độc lập về thống kê).
 *
 * 9) TRAFFIC — THIẾT KẾ LẠI TOÀN BỘ (v4, theo yêu cầu Chương 3/4 luận án):
 *    Bản v3 chỉ sinh 20 luồng (6 TCP + 4 UDP, mỗi chiều), và cả hai loại
 *    đều là lưu lượng "persistent" (BulkSend vô hạn / OnOff hằng số 100%
 *    thời gian ON) — nghĩa là chỉ tạo ra đúng 1 kiểu hành vi lưu lượng
 *    (liên tục, đồng dạng), không đủ đa dạng để tự nhiên sinh ra ba lớp
 *    Mice/Sheep/Elephant theo Bảng 3.2 (Chương 3), và số luồng quá ít để
 *    huấn luyện học liên kết (FRL, cần tối thiểu ~300 luồng).
 *    Bản v4 thay bằng 300 luồng (100 Elephant TCP liên tục + 100 Sheep
 *    TCP-OnOff nhịp vừa + 100 Mice UDP-OnOff nhịp ngắn), xoay vòng qua
 *    toàn bộ 10 nốt client (0-9) và 38 nốt server (12-49) để phân tải đều,
 *    mọi luồng vẫn bắt buộc đi qua bottleneck (10-11) như thiết kế gốc.
 *    Không dùng lại traffic hai chiều (duplex) của v3 để đơn giản hoá quản
 *    lý 300 luồng — chiều ngược của TCP vẫn có ACK đi qua bottleneck theo
 *    đúng cơ chế giao thức, đủ để trace ghi nhận lưu lượng hai chiều ở
 *    mức gói ACK. Chi tiết lý do chọn tham số từng lớp xem chú thích ngay
 *    tại các hàm installElephantFlow/installSheepFlow/installMiceFlow bên
 *    dưới (mục TRAFFIC).
 *
 * 10) SHEEP ON/OFF — RÚT NGẮN CHU KỲ (v5, theo đối chiếu nhãn thực tế):
 *    Sau khi tiền xử lý trace v4 và đối chiếu nhãn gán được (Chương 4) với
 *    cụm traffic theo thiết kế (Chương 3), phát hiện 93,2% gói thiết kế
 *    Sheep bị gán nhãn Elephant — do ON trung bình 2,0s DÀI HƠN cửa sổ gán
 *    nhãn 1s, nên đa số cửa sổ "bắt trúng" giữa lúc luồng đang ON liên tục
 *    suốt cả cửa sổ, trông giống hệt Elephant. Rút ON xuống 0,5s / OFF
 *    xuống 0,15s (chu kỳ 0,65s < 1s cửa sổ) để mỗi cửa sổ bắt được nhiều
 *    hơn 1 chu kỳ ON/OFF trọn vẹn, giúp Duty_cycle/Burstiness đo trong
 *    TỪNG CỬA SỔ phản ánh đúng hành vi ngắt quãng của Sheep. Chi tiết xem
 *    chú thích tại installSheepFlow bên dưới.
 *
 * 11) SHEEP — ĐỔI TỪ TCP SANG UDP OnOff (v6, giải quyết triệt để):
 *    Chạy lại v5 cho kết quả GẦN NHƯ KHÔNG ĐỔI so với v4 (vẫn ~94,0% gói
 *    Sheep bị gán Elephant, thay vì cải thiện như kỳ vọng). Nguyên nhân
 *    thật: tỉ lệ rớt gói của kịch bản khá cao (~18%), khiến TCP của Sheep
 *    liên tục rơi vào retransmission timeout (RTO) — CHÍNH khoảng dừng do
 *    RTO, không phải OffTime cấu hình ở OnOffHelper, mới là thứ chi phối
 *    khoảng lặng quan sát được trên dây (đo trực tiếp: khoảng lặng trung
 *    bình ~2,25s, không khớp với cả hai giả thuyết OffTime 0,7s lẫn 0,15s).
 *    Đổi hẳn Sheep từ TCP OnOff sang UDP OnOff (v6): UDP không có RTO/
 *    retransmission nên lịch ON/OFF được tuân thủ chính xác trên dây, bất
 *    kể mạng nghẽn hay không. Đánh đổi: bản v4-v5 dùng TCP cho Sheep vì
 *    muốn Sheep "có phản ứng tắc nghẽn kiểu TCP, khác Elephant" — nay đặc
 *    điểm đó chỉ còn ở Elephant; Sheep và Mice đều là UDP OnOff, chỉ khác
 *    tốc độ/kích thước gói/nhịp ON-OFF. Chi tiết xem chú thích tại
 *    installSheepFlow bên dưới.
 *
 * 12) THÊM CỘT src_port/dst_port (v7, vá lỗi gộp nhầm luồng ở Python):
 *    Đối chiếu nhãn v6 với cụm traffic thiết kế phát hiện 100/100 cặp
 *    (src_ip, dst_ip) UDP mang CẢ HAI kích cỡ gói 158B (Mice) lẫn 608B
 *    (Sheep) — vì Elephant/Sheep/Mice của cùng 1 chỉ số i dùng CHUNG một
 *    cặp (clientNode, serverNode), chỉ khác cổng, mà file CSV trước đây
 *    KHÔNG ghi lại cổng, khiến process_ns3_colab_v3.py định danh luồng chỉ
 *    bằng (src_ip, dst_ip, protocol) và gộp nhầm các luồng khác lớp thành
 *    1 "luồng" duy nhất — làm hỏng toàn bộ đặc trưng cửa sổ tính trên luồng
 *    gộp sai đó. Đây là nguyên nhân GỐC của phần lớn hiện tượng nhãn không
 *    khớp thiết kế traffic ở các bản trước, không phải do tham số ON/OFF.
 *    Nay peek thêm TcpHeader/UdpHeader lấy số cổng nguồn/đích thật, ghi ra
 *    2 cột src_port/dst_port mới — dùng làm thành phần bắt buộc của flow_id
 *    ở process_ns3_colab_v4.py (script Python đi kèm). Kết quả sau khi vá:
 *    Sheep đạt 97% chính xác (từ 50%). Chi tiết xem chú thích tại
 *    TraceIpHeaderCommon bên dưới.
 *
 * 13) MICE ON/OFF — RÚT NGẮN CHU KỲ (v8, cùng nguyên nhân đã sửa cho Sheep):
 *    Sau khi vá lỗi flow_id thiếu port (mục 12 trên), Sheep đạt 97% chính
 *    xác nhưng Mice vẫn bị gán nhãn Sheep ở 76,2% cửa sổ. Soi trực tiếp:
 *    một cửa sổ Mice bị gán sai có 266 gói trải đều gần suốt cả 1 giây,
 *    không phải 1 đợt ON ngắn như cấu hình. Nguyên nhân: chu kỳ cũ (ON
 *    0,15s + OFF 0,9s = 1,05s) GẦN BẰNG đúng độ dài cửa sổ gán nhãn (1s) —
 *    đúng "điểm rơi tệ nhất" về thống kê đã từng làm hỏng Sheep ở v4/v5:
 *    một cửa sổ có thể ngẫu nhiên bắt trọn 1 đợt ON cộng thêm rìa các đợt
 *    ON liền kề, khiến Duty_cycle đo được dao động lớn và bị đẩy cao hơn
 *    thiết kế. Rút chu kỳ xuống 0,2s (ON 0,03s/OFF 0,17s, ~5 chu kỳ trọn
 *    vẹn/cửa sổ) để Duty_cycle ổn định quanh giá trị thiết kế, giữ nguyên
 *    tỉ lệ ON/OFF (~0,15) nên bản chất lưu lượng thưa của Mice không đổi.
 *    Chi tiết xem chú thích tại installMiceFlow bên dưới.
 * ═══════════════════════════════════════════════════════════════════════
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ipv4-queue-disc-item.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"
#include "ns3/ppp-header.h"

#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <tuple>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SimulationLA50v2");

// ================= CSV (DUY NHẤT — hợp nhất IP header + queue + RTT) =================
std::ofstream csv;

// RTT toàn cục gần nhất quan sát được từ tầng TCP (xem ghi chú tại RttTraceHandler
// bên dưới) — khai báo sớm vì TraceIpHeaderCommon cần đọc giá trị này.
double g_lastRttMs = 0.0;

// Khóa định danh 1 gói tin xuyên suốt các hop: (src_ip, dst_ip, identification)
using PacketKey = std::tuple<uint32_t, uint32_t, uint16_t>;

static PacketKey MakeKey(const Ipv4Header &ipHeader)
{
    return std::make_tuple(ipHeader.GetSource().Get(),
                            ipHeader.GetDestination().Get(),
                            ipHeader.GetIdentification());
}

// ================= TRACE IP HEADER =================
// Hàm dùng chung, ghi 1 dòng CSV cho cả sự kiện TX lẫn RX (event_type truyền vào)
// Bổ sung: nếu là gói TCP, peek thêm TcpHeader để lấy cờ ACK -> cột tcp_ack_flag
// (0/1), và số thứ tự (Sequence Number) -> cột tcp_seq, phục vụ tính ACK Ratio
// và Retransmission Rate thật (công thức 4.4, mục 4.1.1.3) thay vì để NaN.
// VÁ (bản v4): trước đây chỉ lấy cờ ACK, KHÔNG lấy Sequence Number, nên không
// thể phát hiện gói truyền lại (retransmission) ở bước tiền xử lý Python —
// mọi số thứ tự TCP trùng lặp trong luồng (đã loại trừ ACK thuần không mang
// dữ liệu) đều được tính là một lần truyền lại.
//
// VÁ (v7) — LỖI ĐỊNH DANH LUỒNG Ở PYTHON: đối chiếu nhãn thực tế phát hiện
// đúng 100/100 cặp (src_ip, dst_ip) UDP mang CẢ HAI kích cỡ gói 158B (Mice)
// LẪN 608B (Sheep) — vì thiết kế 300 luồng (mục TRAFFIC bên dưới) cho
// Elephant/Sheep/Mice của cùng 1 chỉ số i dùng CHUNG một cặp (clientNode,
// serverNode), chỉ khác PORT. Do file CSV trước đây KHÔNG ghi lại port, các
// script Python (process_ns3_colab_v3.py) buộc phải định danh luồng chỉ
// bằng (src_ip, dst_ip, protocol) -> gộp NHẦM 2-3 luồng khác lớp có cùng
// cặp IP/giao thức thành 1 "luồng" duy nhất, làm hỏng toàn bộ đặc trưng cửa
// sổ (volume_window, duty_cycle...) tính trên luồng gộp sai đó. Đây mới là
// nguyên nhân GỐC của hiện tượng nhãn không khớp thiết kế traffic, không
// phải do tham số ON/OFF hay TCP retransmission như các giả thuyết trước.
// Nay peek thêm TcpHeader/UdpHeader để lấy số cổng nguồn/đích thật -> cột
// src_port/dst_port, dùng làm thành phần bắt buộc của flow_id ở Python.
void TraceIpHeaderCommon(const std::string &eventType,
                          Ptr<const Packet> packet)
{
    Ptr<Packet> copy = packet->Copy();
    Ipv4Header ipHeader;

    if (copy->PeekHeader(ipHeader))
    {
        bool DF = ipHeader.IsDontFragment();
        bool MF = !ipHeader.IsLastFragment();

        int ackFlag = -1;         // -1 = không áp dụng (không phải TCP)
        int64_t seqNum = -1;      // -1 = không áp dụng (không phải TCP)
        int32_t srcPort = -1;     // -1 = không xác định được (không phải TCP/UDP)
        int32_t dstPort = -1;

        if ((uint32_t)ipHeader.GetProtocol() == 6)   // 6 = TCP
        {
            Ptr<Packet> afterIp = copy->Copy();
            afterIp->RemoveHeader(ipHeader);
            TcpHeader tcpHeader;
            if (afterIp->PeekHeader(tcpHeader))
            {
                ackFlag = (tcpHeader.GetFlags() & TcpHeader::ACK) ? 1 : 0;
                seqNum = static_cast<int64_t>(tcpHeader.GetSequenceNumber().GetValue());
                srcPort = tcpHeader.GetSourcePort();
                dstPort = tcpHeader.GetDestinationPort();
            }
        }
        else if ((uint32_t)ipHeader.GetProtocol() == 17)   // 17 = UDP
        {
            Ptr<Packet> afterIp = copy->Copy();
            afterIp->RemoveHeader(ipHeader);
            UdpHeader udpHeader;
            if (afterIp->PeekHeader(udpHeader))
            {
                srcPort = udpHeader.GetSourcePort();
                dstPort = udpHeader.GetDestinationPort();
            }
        }

        csv
            << Simulator::Now().GetSeconds() << ","
            << eventType << ","
            << ","                                   // direction (chỉ áp dụng cho ENQUEUE/DEQUEUE/DROP)
            << ipHeader.GetSource() << ","
            << ipHeader.GetDestination() << ","
            << (uint32_t)ipHeader.GetProtocol() << ","
            << (uint32_t)ipHeader.GetTos() << ","
            << (uint32_t)ipHeader.GetTtl() << ","
            << (uint32_t)ipHeader.GetIdentification() << ","
            << ipHeader.GetFragmentOffset() << ","
            << MF << ","
            << DF << ","
            << ipHeader.GetPayloadSize() << ","
            << ackFlag << ","
            << seqNum << ","                          // số thứ tự TCP (rỗng/-1 nếu UDP)
            << srcPort << ","                          // MỚI (v7): cổng nguồn (-1 nếu không phải TCP/UDP)
            << dstPort << ","                          // MỚI (v7): cổng đích
            << "," << ","                            // queue_size, d_queue_ms (không áp dụng)
            << "," << g_lastRttMs                     // rtt_context (bỏ trống), rtt_ms (snapshot gần nhất)
            << std::endl;
    }
}

// Wrapper khớp đúng chữ ký (signature) của trace source Ipv4L3Protocol/Tx
void TraceIpHeaderTx(std::string context,
                      Ptr<const Packet> packet,
                      Ptr<Ipv4> ipv4,
                      uint32_t interface)
{
    TraceIpHeaderCommon("TX", packet);
}

// Wrapper khớp đúng chữ ký của trace source Ipv4L3Protocol/Rx
// (Rx trace nguồn có thêm interface đến — cùng kiểu tham số nên dùng chung được)
void TraceIpHeaderRx(std::string context,
                      Ptr<const Packet> packet,
                      Ptr<Ipv4> ipv4,
                      uint32_t interface)
{
    TraceIpHeaderCommon("RX", packet);
}

// ================= TRACE HÀNG ĐỢI TẠI BOTTLENECK =================
// Ghi lại ENQUEUE / DEQUEUE / DROP thật của hàng đợi bottleneck (cả 2 chiều),
// hoạt động với CẢ DropTailQueue<Packet> (chế độ droptail) LẪN QueueDisc
// (chế độ codel/fqcodel/pie/cake), cho ra queue_size và d_queue THẬT thay vì NaN.
// Ghi vào CÙNG file CSV duy nhất với TX/RX (không tách file riêng), theo
// đúng thứ tự 21 cột hợp nhất — các cột không áp dụng cho loại dòng này
// (protocol,tos,ttl,identification,fragment_offset,MF,DF,payload_size,
// tcp_ack_flag,tcp_seq,src_port,dst_port) để trống. tcp_seq/src_port/dst_port
// để trống ở đây (không phải vì không áp dụng được về nguyên tắc, mà vì
// WriteQueueRow chỉ nhận Ipv4Header, không có TCP/UDP header — Retransmission
// Rate và định danh luồng theo cổng chỉ cần tính từ các dòng TX là đủ).
std::map<PacketKey, double> g_enqueueTime;   // key -> thời điểm vào hàng đợi (s)
uint32_t g_queueOcc[2] = {0, 0};             // occupancy hiện tại: [0]=N11->N12, [1]=N12->N11

static void WriteQueueRow(int dir, const std::string &evt, const Ipv4Header &ipHeader, double dQueueMs)
{
    csv
        << Simulator::Now().GetSeconds() << ","
        << evt << ","
        << dir << ","
        << ipHeader.GetSource() << ","
        << ipHeader.GetDestination() << ","
        << "," << "," << ","                      // protocol, tos, ttl (không áp dụng)
        << (uint32_t)ipHeader.GetIdentification() << ","
        << "," << "," << ","                       // fragment_offset, MF, DF (de trong)
        << ipHeader.GetPayloadSize() << ","        // v6: payload_size GHI TRUC TIEP tai ENQUEUE/DEQUEUE/DROP
                                                    // (cung dinh nghia voi dong TX) - khong can ghep TX
                                                    // gan nhat nua (ghep sai khi AQM xep lai thu tu goi: FQ-CoDel, CAKE)
        << "," << ","                               // tcp_ack_flag, tcp_seq
        << "," << ","                               // MỚI (v7): src_port, dst_port
        << g_queueOcc[dir] << ","
        << dQueueMs << ","
        << "," << g_lastRttMs                       // rtt_context (bỏ trống), rtt_ms (snapshot)
        << std::endl;
}

// Đọc IPv4 header từ gói tin nằm TRONG hàng đợi thiết bị (device Queue<Packet>).
// Gói ở tầng này đã được PointToPointNetDevice đóng gói thêm PPP header (2 byte)
// phía trước, KHÁC với gói ở tầng Ipv4L3Protocol (TraceIpHeaderCommon) vốn còn
// thuần IP. Thử đọc thẳng IP header trước; nếu thất bại (do dính PPP header),
// bóc PPP header rồi thử lại — cách này hoạt động đúng cho cả hai chế độ:
//   - droptail: gói NẰM SAU PPP header (thử lần 2 sẽ thành công)
//   - AQM (QueueDisc): gói thường vẫn thuần IP (thử lần 1 đã thành công)
static bool TryPeekIpv4Header(Ptr<const Packet> packet, Ipv4Header &ipHeader)
{
    Ptr<Packet> copy = packet->Copy();
    if (copy->PeekHeader(ipHeader))
    {
        return true;
    }
    copy = packet->Copy();
    PppHeader ppp;
    copy->RemoveHeader(ppp);
    return copy->PeekHeader(ipHeader);
}

void QueueEnqueueHandler(int dir, Ptr<const Packet> packet)
{
    g_queueOcc[dir]++;
    Ipv4Header ipHeader;
    if (TryPeekIpv4Header(packet, ipHeader))
    {
        g_enqueueTime[MakeKey(ipHeader)] = Simulator::Now().GetSeconds();
        WriteQueueRow(dir, "ENQUEUE", ipHeader, 0.0);
    }
}

void QueueDequeueHandler(int dir, Ptr<const Packet> packet)
{
    if (g_queueOcc[dir] > 0) g_queueOcc[dir]--;
    Ipv4Header ipHeader;
    if (TryPeekIpv4Header(packet, ipHeader))
    {
        double dQueueMs = 0.0;
        auto it = g_enqueueTime.find(MakeKey(ipHeader));
        if (it != g_enqueueTime.end())
        {
            dQueueMs = (Simulator::Now().GetSeconds() - it->second) * 1000.0;
            g_enqueueTime.erase(it);
        }
        WriteQueueRow(dir, "DEQUEUE", ipHeader, dQueueMs);
    }
}

void QueueDropHandler(int dir, Ptr<const Packet> packet)
{
    // Gói bị rớt không rời hàng đợi qua Dequeue nên phải tự trừ occupancy ở đây
    if (g_queueOcc[dir] > 0) g_queueOcc[dir]--;
    Ipv4Header ipHeader;
    if (TryPeekIpv4Header(packet, ipHeader))
    {
        g_enqueueTime.erase(MakeKey(ipHeader));
        WriteQueueRow(dir, "DROP", ipHeader, 0.0);
    }
}

// ---- Wrapper cho QueueDisc (che do AQM: codel/fqcodel/pie/cake/red/ared) ----
// QueueDisc dung Ptr<const QueueDiscItem> thay vi Ptr<const Packet>.
//
// SUA LOI (quan trong): KHONG duoc goi item->GetPacket() roi PeekHeader<Ipv4Header>
// nhu QueueEnqueueHandler/QueueDequeueHandler/QueueDropHandler (danh cho DropTail)
// lam - voi QueueDiscItem, IP header CHUA duoc serialize vao buffer goi tin, ma
// luu rieng thanh 1 object ben trong Ipv4QueueDiscItem (chi duoc AddHeader() luc
// QueueDisc thuc su gui goi di). Goi PeekHeader truc tiep se doc nham byte dau
// cua phan payload TCP/UDP thanh IP header, sinh ra src_ip/dst_ip SAI HOAN TOAN
// (vi du "64.0.0.0"/"0.0.8.10" quan sat duoc trong du lieu cu) - khong lam sai
// Timestamp/queue_size/d_queue_ms (khong phu thuoc IP header) nhung lam SAI moi
// chi tieu can nhan dang luong theo IP (Thong luong qua DEQUEUE, Jain's Index).
//
// Cach dung: DynamicCast sang Ipv4QueueDiscItem va goi ->GetHeader() de lay
// dung Ipv4Header da luu san, khong can (va khong duoc) peek tu packet buffer.
static bool TryGetIpv4HeaderFromQueueDiscItem(Ptr<const QueueDiscItem> item, Ipv4Header &ipHeader)
{
    Ptr<const Ipv4QueueDiscItem> ipItem = DynamicCast<const Ipv4QueueDiscItem>(item);
    if (ipItem)
    {
        ipHeader = ipItem->GetHeader();
        return true;
    }
    return false;
}

void QueueDiscEnqueueHandler(int dir, Ptr<const QueueDiscItem> item)
{
    g_queueOcc[dir]++;
    Ipv4Header ipHeader;
    if (TryGetIpv4HeaderFromQueueDiscItem(item, ipHeader))
    {
        g_enqueueTime[MakeKey(ipHeader)] = Simulator::Now().GetSeconds();
        WriteQueueRow(dir, "ENQUEUE", ipHeader, 0.0);
    }
}
void QueueDiscDequeueHandler(int dir, Ptr<const QueueDiscItem> item)
{
    if (g_queueOcc[dir] > 0) g_queueOcc[dir]--;
    Ipv4Header ipHeader;
    if (TryGetIpv4HeaderFromQueueDiscItem(item, ipHeader))
    {
        double dQueueMs = 0.0;
        auto it = g_enqueueTime.find(MakeKey(ipHeader));
        if (it != g_enqueueTime.end())
        {
            dQueueMs = (Simulator::Now().GetSeconds() - it->second) * 1000.0;
            g_enqueueTime.erase(it);
        }
        WriteQueueRow(dir, "DEQUEUE", ipHeader, dQueueMs);
    }
}
void QueueDiscDropHandler(int dir, Ptr<const QueueDiscItem> item)
{
    if (g_queueOcc[dir] > 0) g_queueOcc[dir]--;
    Ipv4Header ipHeader;
    if (TryGetIpv4HeaderFromQueueDiscItem(item, ipHeader))
    {
        g_enqueueTime.erase(MakeKey(ipHeader));
        WriteQueueRow(dir, "DROP", ipHeader, 0.0);
    }
}

// ================= TRACE RTT TCP THẬT =================
// Hook trực tiếp bộ ước lượng RTT của TCP tại tầng socket. Lưu ý: trace này
// KHÔNG cho biết trực tiếp src/dst IP trong tham số callback, chỉ có
// "context" (đường dẫn node/socket trong Config namespace) — nên KHÔNG gán
// được chính xác cho 1 luồng cụ thể. Cách xử lý:
//   (1) Cập nhật biến toàn cục g_lastRttMs = RTT gần nhất quan sát được trên
//       TOÀN mạng — được đính kèm (snapshot) vào MỌI dòng TX/RX/ENQUEUE/
//       DEQUEUE/DROP ghi ra sau đó (xem g_lastRttMs ở trên).
//   (2) Đồng thời ghi 1 dòng riêng event_type=RTT_UPDATE kèm context gốc,
//       để nếu cần đối chiếu chính xác hơn theo từng node/socket ở bước
//       xử lý Python vẫn còn dữ liệu thô đầy đủ.
void RttTraceHandler(std::string context, Time oldRtt, Time newRtt)
{
    g_lastRttMs = newRtt.GetMilliSeconds();

    csv
        << Simulator::Now().GetSeconds() << ","
        << "RTT_UPDATE" << ","
        << "," << "," << ","                       // direction, src_ip, dst_ip (không áp dụng)
        << "," << "," << "," << ","                 // protocol, tos, ttl, identification
        << "," << "," << "," << ","                 // fragment_offset, MF, DF, payload_size
        << "," << ","                                // tcp_ack_flag, tcp_seq
        << "," << ","                                // MỚI (v7): src_port, dst_port
        << "," << ","                                // queue_size, d_queue_ms
        << context << "," << g_lastRttMs
        << std::endl;
}

// ================= MAIN =================
int main(int argc, char *argv[])
{
    // ── (8) SEED — mặc định = 10, đặt NGAY ĐẦU main(), trước khi khởi tạo
    //     bất kỳ đối tượng NS-3 nào dùng đến RNG. RunNumber giữ mặc định 1,
    //     dùng "--run=1..5" (giữ nguyên seed=10) để chạy n=5 lần lặp độc lập
    //     (independent replications theo khuyến nghị của ns-3) phục vụ tính
    //     độ lệch chuẩn giữa các lần chạy.
    uint32_t seed = 10;
    uint32_t run  = 1;
    std::string qdisc = "droptail";   // droptail | codel | fqcodel | pie | cake

    CommandLine cmd;
    cmd.AddValue("seed",  "RNG seed", seed);
    cmd.AddValue("run",   "RNG run number", run);
    cmd.AddValue("qdisc", "Co che hang doi tai bottleneck: droptail|codel|fqcodel|pie|cake|red|ared", qdisc);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    double simTime = 180.0;

    // ── NHIỄU THỜI GIAN KHỞI ĐỘNG (start jitter) — BẮT BUỘC để n lần chạy độc
    // lập (--run=1..5) thực sự khác nhau về mặt thống kê. Nếu không có biến
    // ngẫu nhiên nào được "rút" trong toàn bộ mô phỏng (BulkSend/OnOff hằng
    // số/Start cố định đều tất định), thì RngRun không tạo ra khác biệt gì —
    // mọi luồng đều lệch đúng cùng 1 giá trị. Jitter nhỏ (0-500ms) không làm
    // thay đổi bản chất kịch bản, chỉ đủ để "tiêu thụ" RNG stream theo đúng
    // seed/run, giúp mỗi lần chạy độc lập có sai khác thực sự.
    Ptr<UniformRandomVariable> startJitter = CreateObject<UniformRandomVariable>();
    startJitter->SetAttribute("Min", DoubleValue(0.0));
    startJitter->SetAttribute("Max", DoubleValue(0.5));

    // ── (5) Giới hạn cửa sổ TCP = 16KB — set TRƯỚC khi tạo app/socket ──
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(16384));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(16384));

    // ── VÁ (phát hiện từ file trace thật): mặc định ns3::TcpSocket::SegmentSize
    // = 536 byte (giá trị lịch sử theo RFC 879), khiến MỌI gói TCP trên dây bị
    // cắt xuống tối đa ~568B (536 data + 32B TCP header có timestamp option) —
    // BẤT KỂ SendSize của BulkSendHelper (Elephant, 1200B) hay PacketSize của
    // OnOffHelper (Sheep, 600B) đặt là bao nhiêu. Hệ quả: Elephant và Sheep
    // không còn phân biệt được qua Packet_size (Bảng 3.2) vì cả hai đều bị ép
    // về cùng ~568B. Set SegmentSize=1400 (dưới MTU Ethernet mặc định 1500,
    // đủ chỗ cho IP+TCP header) để gói Elephant (1200B) đi nguyên vẹn không bị
    // phân mảnh, đồng thời Sheep (600B) vẫn nhỏ hơn MSS nên cũng không bị cắt.
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1400));

    NodeContainer nodes;
    nodes.Create(50);

    InternetStackHelper stack;
    stack.Install(nodes);

    // ================= LINKS =================
    PointToPointHelper access;
    access.SetDeviceAttribute("DataRate", StringValue("20Mbps"));
    access.SetChannelAttribute("Delay", StringValue("5ms"));

    PointToPointHelper bottleneck;
    bottleneck.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    bottleneck.SetChannelAttribute("Delay", StringValue("50ms"));

    // ── (3) Hàng đợi tại bottleneck — DropTail (baseline) hoặc AQM ──
    // Chọn qua --qdisc=droptail|codel|fqcodel|pie|cake|red|ared (mặc định droptail).
    // Ngân sách hàng đợi thống nhất ~300 gói cho MỌI phương án để so sánh công bằng.
    //
    // dCoDel và DN-dCoD (thuật toán đề xuất của luận án) KHÔNG có trong danh
    // sách này — theo quyết định của NCS, hai cơ chế này được xử lý OFFLINE
    // trên Python (mục 4.2.3), tái sử dụng trực tiếp vết lưu lượng (trace)
    // sinh ra từ đây (đặc biệt là trace của DropTail, làm "nền trung lập"),
    // KHÔNG cài đặt thành QueueDisc thật trong NS-3 ở giai đoạn này.
    //
    // LƯU Ý: ns-3 bản chuẩn (mainline) chưa có lớp "CakeQueueDisc" riêng biệt.
    // FqCobaltQueueDisc là thành phần AQM lõi của CAKE (thuật toán COBALT kết hợp
    // bộ lập lịch FlowQueue) đã được port sang ns-3 (dựa trên sch_cake.c của Linux
    // kernel), nên được dùng làm đại diện gần nhất cho CAKE trong mô phỏng này.
    // Nếu hội đồng hỏi, cần nêu rõ đây là "CAKE-equivalent (FqCobalt)", không phải
    // bản CAKE đầy đủ 1:1 như trong Linux.
    access.DisableFlowControl();   // access link giữ nguyên DropTail thuần, không đổi

    TrafficControlHelper tchBottleneck;
    bool useAqm = (qdisc != "droptail");

    if (qdisc == "droptail")
    {
        bottleneck.SetQueue("ns3::DropTailQueue<Packet>",
                             "MaxSize", QueueSizeValue(QueueSize("300p")));
        bottleneck.DisableFlowControl();
    }
    else if (qdisc == "codel")
    {
        // CoDel GỐC (RFC 8289) — dùng làm baseline "cổ điển dựa trên độ trễ
        // mục tiêu cố định" trong Bảng so sánh AQM (Thực nghiệm 3, mục
        // 4.2.3), đối chứng trực tiếp cho dCoDel (biến thể động của chính
        // luận án). Target/Interval đặt tường minh đúng giá trị mặc định
        // kinh điển của CoDel (5ms/100ms) để dễ trích dẫn trong luận án,
        // dù đây cũng là giá trị mặc định sẵn có của CoDelQueueDisc trong ns-3.
        tchBottleneck.SetRootQueueDisc("ns3::CoDelQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")),
                                        "Target", TimeValue(MilliSeconds(5)),
                                        "Interval", TimeValue(MilliSeconds(100)));
    }
    else if (qdisc == "fqcodel")
    {
        tchBottleneck.SetRootQueueDisc("ns3::FqCoDelQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")));
    }
    else if (qdisc == "pie")
    {
        tchBottleneck.SetRootQueueDisc("ns3::PieQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")));
    }
    else if (qdisc == "cake")
    {
        tchBottleneck.SetRootQueueDisc("ns3::FqCobaltQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")));
    }
    else if (qdisc == "red")
    {
        tchBottleneck.SetRootQueueDisc("ns3::RedQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")));
    }
    else if (qdisc == "ared")
    {
        // SUA LOI: ns-3 KHONG co class rieng "ns3::AredQueueDisc" (khong
        // ton tai trong module traffic-control) - ARED chi la 1 THUOC TINH
        // (attribute "ARED") bat tren CHINH ns3::RedQueueDisc, tu dong cau
        // hinh Queue weight/MinTh/MaxTh va thich nghi xac suat tha goi toi
        // da (m_curMaxP) - dung nhu tai lieu chinh thuc cua ns-3
        // (src/traffic-control/examples/adaptive-red-tests.cc):
        //   Config::SetDefault("ns3::RedQueueDisc::ARED", BooleanValue(true));
        // Day chinh la nguyen nhan gay loi "NS_ASSERT failed... TypeId::
        // LookupByName: ns3::AredQueueDisc not found" khi chay --qdisc=ared.
        tchBottleneck.SetRootQueueDisc("ns3::RedQueueDisc",
                                        "MaxSize", QueueSizeValue(QueueSize("300p")),
                                        "ARED", BooleanValue(true));
    }
    else
    {
        std::cerr << "[CANH BAO] --qdisc=" << qdisc
                  << " khong hop le (droptail|codel|fqcodel|pie|cake|red|ared). Dung droptail mac dinh.\n";
        bottleneck.SetQueue("ns3::DropTailQueue<Packet>",
                             "MaxSize", QueueSizeValue(QueueSize("300p")));
        bottleneck.DisableFlowControl();
        useAqm = false;
        qdisc = "droptail";
    }

    std::vector<NetDeviceContainer> devs;

    // 0–9 -> node 10  (N1–N10 -> N11)
    for (int i = 0; i < 10; i++)
        devs.push_back(access.Install(nodes.Get(i), nodes.Get(10)));

    // bottleneck 10–11 (N11–N12)
    NetDeviceContainer bottleneckDev =
        bottleneck.Install(nodes.Get(10), nodes.Get(11));

    // Cài queue disc AQM (nếu có) lên bottleneckDev NGAY sau khi tạo device,
    // PHẢI làm trước khi gán địa chỉ IP (addr.Assign) theo đúng quy tắc ns-3.
    QueueDiscContainer bottleneckQdiscs;
    if (useAqm)
    {
        bottleneckQdiscs = tchBottleneck.Install(bottleneckDev);

        // dir 0 = hàng đợi thuộc bottleneckDev.Get(0) (chiều N11 -> N12)
        // dir 1 = hàng đợi thuộc bottleneckDev.Get(1) (chiều N12 -> N11)
        for (int dir = 0; dir < 2; dir++)
        {
            Ptr<QueueDisc> qd = bottleneckQdiscs.Get(dir);
            qd->TraceConnectWithoutContext("Enqueue", MakeBoundCallback(&QueueDiscEnqueueHandler, dir));
            qd->TraceConnectWithoutContext("Dequeue", MakeBoundCallback(&QueueDiscDequeueHandler, dir));
            qd->TraceConnectWithoutContext("Drop", MakeBoundCallback(&QueueDiscDropHandler, dir));
        }
    }
    else
    {
        for (int dir = 0; dir < 2; dir++)
        {
            Ptr<PointToPointNetDevice> p2p =
                DynamicCast<PointToPointNetDevice>(bottleneckDev.Get(dir));
            Ptr<Queue<Packet>> q = p2p->GetQueue();
            q->TraceConnectWithoutContext("Enqueue", MakeBoundCallback(&QueueEnqueueHandler, dir));
            q->TraceConnectWithoutContext("Dequeue", MakeBoundCallback(&QueueDequeueHandler, dir));
            q->TraceConnectWithoutContext("DropBeforeEnqueue", MakeBoundCallback(&QueueDropHandler, dir));
        }
    }

    // 12–49 -> node 11 (N13–N50 -> N12)
    for (int i = 12; i < 50; i++)
        devs.push_back(access.Install(nodes.Get(i), nodes.Get(11)));

    // ================= IP =================
    Ipv4AddressHelper addr;
    std::vector<Ipv4InterfaceContainer> ifs;

    int subnet = 1;
    for (auto &d : devs)
    {
        std::ostringstream net;
        net << "10.1." << subnet++ << ".0";
        addr.SetBase(net.str().c_str(), "255.255.255.0");
        ifs.push_back(addr.Assign(d));
    }

    addr.SetBase("10.1.100.0", "255.255.255.0");
    Ipv4InterfaceContainer bottleneckIf = addr.Assign(bottleneckDev);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Map từ "devs index" -> node index thật, để tra cứu IP theo đúng nốt
    // (devs[0..9] ứng với nốt 0..9 ; devs[10..47] ứng với nốt 12..49).
    // (Lambda ifaceOfClientNode của bản v3 đã bỏ ở v4 vì không còn traffic
    // hai chiều dùng đến nó — nếu sau này cần thêm luồng phản hồi thủ công
    // từ phía server về client, có thể khôi phục lambda này.)
    auto ifaceOfServerNode = [&](int nodeIdx) -> Ipv4Address {
        // nodeIdx trong [12,49] -> devs[10 + (nodeIdx-12)]
        int devIdx = 10 + (nodeIdx - 12);
        return ifs[devIdx].GetAddress(0);
    };

    // ================= TRAFFIC (v4 — 300 luồng, 3 lớp Mice/Sheep/Elephant) =====
    // Thay cho bộ traffic cũ (6 TCP + 4 UDP, hai chiều, 20 luồng), phiên bản
    // này sinh NUM_FLOWS_PER_CLASS * 3 = 300 luồng độc lập, chia đều ba "hồ
    // sơ" lưu lượng tương ứng với ba lớp mục tiêu ở Bảng 3.2 (Chương 3):
    //   (a) đủ số luồng tối thiểu cho FRL (>= 300 luồng theo yêu cầu);
    //   (b) tự nhiên sinh ra phân bố ba lớp cân bằng hơn (thay vì chỉ có
    //       một kiểu lưu lượng "persistent" đồng dạng như bản cũ), giúp bộ
    //       ngưỡng tam phân vị ở mục 3.1.1.2 tách lớp có ý nghĩa thay vì
    //       suy biến về một, hai lớp.
    //
    // Cả ba lớp đều dùng client ở nốt 0-9 (N1-N10) và server ở nốt 12-49
    // (N13-N50), xoay vòng (round-robin) qua toàn bộ 10 nốt client và 38
    // nốt server để phân tải đồng đều — NÊN MỌI GÓI TIN vẫn bắt buộc đi
    // qua bottleneck (10-11) đúng như thiết kế gốc. Mỗi luồng có cổng
    // (port) riêng, không trùng giữa các luồng.
    //
    // GHI CHÚ MÔ HÌNH HOÁ (nêu rõ để giải trình khi bảo vệ): gói tin trong
    // mỗi luồng OnOff (Sheep/Mice) có kích thước CỐ ĐỊNH trong nội bộ luồng
    // đó (giới hạn của OnOffHelper chuẩn NS-3, không hỗ trợ kích thước gói
    // ngẫu nhiên theo từng gói), nên Packet_size_CV nội-luồng sẽ gần 0 —
    // không hoàn toàn khớp ngưỡng "CV cao" gán cho Mice trong Bảng 3.2,
    // nhưng phản ánh đúng hành vi THỰC TẾ của lưu lượng điều khiển đồng
    // dạng (DNS/ICMP). Bốn đặc trưng còn lại (Volume_window, Duty_cycle,
    // Burstiness, Peak_rate) đã đủ khác biệt giữa ba lớp để ngưỡng tam
    // phân vị (tính lại theo đúng phân bố thực tế của bộ dữ liệu mô phỏng
    // này, không dùng cứng số liệu tuyệt đối trong Bảng 3.2) tách lớp có
    // ý nghĩa thống kê.

    const int NUM_FLOWS_PER_CLASS = 100;   // 100 Mice + 100 Sheep + 100 Elephant = 300 luồng
    const int NUM_CLIENT_NODES    = 10;    // nốt 0..9   (N1..N10)
    const int NUM_SERVER_NODES    = 38;    // nốt 12..49 (N13..N50)

    // Tốc độ khi ON của Sheep/Mice được rút ngẫu nhiên trong một khoảng để
    // các luồng cùng lớp không hoàn toàn giống hệt nhau (tránh mô hình học
    // được một hằng số duy nhất thay vì học đặc trưng phân biệt lớp thật).
    Ptr<UniformRandomVariable> sheepRateRv = CreateObject<UniformRandomVariable>();
    sheepRateRv->SetAttribute("Min", DoubleValue(30.0));   // kbps
    sheepRateRv->SetAttribute("Max", DoubleValue(80.0));   // kbps

    Ptr<UniformRandomVariable> miceRateRv = CreateObject<UniformRandomVariable>();
    miceRateRv->SetAttribute("Min", DoubleValue(40.0));    // kbps
    miceRateRv->SetAttribute("Max", DoubleValue(120.0));   // kbps

    // ---- Elephant: TCP BulkSend, không giới hạn dung lượng, chạy gần suốt
    // 180s -> Duty_cycle rất cao (kỳ vọng > 0,9), khối lượng byte lớn, kích
    // thước gói đồng dạng (SendSize 1200B, vượt ngưỡng Elephant 1008B ở
    // Bảng 3.2). TCP (không phải UDP) để hành vi "chiếm băng thông tối đa
    // có thể" phản ánh đúng bản chất co giãn (elastic) của Elephant thật.
    auto installElephantFlow = [&](int clientNode, int serverNode, uint16_t port)
    {
        PacketSinkHelper sink("ns3::TcpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
        sink.Install(nodes.Get(serverNode));

        BulkSendHelper tcp("ns3::TcpSocketFactory",
                            InetSocketAddress(ifaceOfServerNode(serverNode), port));
        tcp.SetAttribute("SendSize", UintegerValue(1200));
        tcp.SetAttribute("MaxBytes", UintegerValue(0));   // không giới hạn dung lượng

        auto app = tcp.Install(nodes.Get(clientNode));
        app.Start(Seconds(1.0 + startJitter->GetValue()));
        app.Stop(Seconds(simTime));
    };

    // ---- Sheep: UDP OnOff, tốc độ vừa (30-80 kbps khi ON), ON trung bình
    // 0,5s / OFF trung bình 0,15s -> Duty_cycle kỳ vọng xấp xỉ 0,77 (nằm
    // trong khoảng 0,6-0,9 của Bảng 3.2).
    //
    // VÁ (v5, phát hiện từ đối chiếu nhãn thực tế trên trace): bản v4 dùng
    // TCP OnOff với ON trung bình 2,0s / OFF 0,7s -> chu kỳ 2,7s, DÀI HƠN
    // cửa sổ gán nhãn 1s, khiến 93,2% gói thiết kế Sheep bị gán nhãn
    // Elephant. Rút ON xuống 0,5s / OFF 0,15s (v5) KHÔNG cải thiện được gì
    // (vẫn ~94,0% bị gán Elephant) - vì tỉ lệ rớt gói của toàn kịch bản khá
    // cao (~18%), TCP của Sheep liên tục rơi vào retransmission timeout
    // (RTO mặc định NS-3 bắt đầu từ 1s, backoff gấp đôi mỗi lần thất bại),
    // và chính khoảng dừng do RTO - KHÔNG PHẢI OffTime cấu hình ở tầng ứng
    // dụng - mới là thứ chi phối khoảng lặng quan sát được trên dây. Đo
    // trực tiếp trên trace: khoảng lặng trung bình giữa các gói Sheep là
    // ~2,25s (median ~0,84s), không khớp với CẢ HAI giả thuyết OffTime
    // (0,7s lẫn 0,15s) - đúng dấu hiệu bị chi phối bởi RTO chứ không phải
    // lịch OnOff cấu hình.
    //
    // VÁ (v6): đổi hẳn Sheep từ TCP sang UDP OnOff (cùng cơ chế với Mice,
    // khác tham số tốc độ/kích thước gói/ON-OFF). UDP không có cơ chế
    // retransmission/RTO, nên lịch ON/OFF của OnOffHelper được tuân thủ
    // CHÍNH XÁC trên dây bất kể mạng có nghẽn hay không - loại bỏ hoàn
    // toàn nguồn nhiễu đã gây sai lệch ở v4/v5. Đánh đổi: mất đặc điểm
    // "Sheep có phản ứng tắc nghẽn kiểu TCP" - nay chỉ còn Elephant đại
    // diện cho hành vi co giãn TCP, Sheep và Mice đều là UDP OnOff (chỉ
    // khác tốc độ/kích thước gói/nhịp ON-OFF).
    auto installSheepFlow = [&](int clientNode, int serverNode, uint16_t port)
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
        sink.Install(nodes.Get(serverNode));

        OnOffHelper sheep("ns3::UdpSocketFactory",
                           InetSocketAddress(ifaceOfServerNode(serverNode), port));
        uint32_t rateKbps = static_cast<uint32_t>(sheepRateRv->GetValue());
        sheep.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string(rateKbps) + "kbps")));
        sheep.SetAttribute("PacketSize", UintegerValue(600));   // trong khoảng Sheep 372-1008B
        sheep.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
        sheep.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.15]"));

        auto app = sheep.Install(nodes.Get(clientNode));
        app.Start(Seconds(1.0 + startJitter->GetValue()));
        app.Stop(Seconds(simTime));
    };

    // ---- Mice: UDP OnOff, tốc độ thấp (40-120 kbps khi ON), gói nhỏ
    // (150B, dưới ngưỡng Mice 372B), ON trung bình 0,03s / OFF trung bình
    // 0,17s -> chu kỳ 0,2s, Duty_cycle kỳ vọng xấp xỉ 0,15 (< 0,6 của Bảng
    // 3.2), mô phỏng các phiên trao đổi ngắn kiểu DNS/ICMP/TCP control.
    //
    // VÁ (v8, phát hiện từ đối chiếu nhãn thực tế sau khi sửa lỗi flow_id
    // thiếu port ở v7): sau khi Sheep đã tách đúng luồng (97% chính xác),
    // Mice vẫn bị gán nhãn Sheep ở đa số cửa sổ (76,2%). Soi trực tiếp một
    // cửa sổ bị gán sai: 266 gói trải đều gần suốt cả 1 giây — không phải
    // 1 đợt ON ngắn 0,15s như cấu hình cũ. Nguyên nhân: chu kỳ ON+OFF cũ
    // (0,15+0,9=1,05s) GẦN BẰNG đúng độ dài cửa sổ gán nhãn (1s) — đúng
    // "điểm rơi tệ nhất" về thống kê: một cửa sổ có thể ngẫu nhiên bắt trọn
    // 1 đợt ON cộng thêm rìa của các đợt ON liền kề, khiến Duty_cycle đo
    // được dao động rất lớn và thường bị đẩy cao hơn hẳn tỉ lệ thiết kế —
    // ĐÚNG CÙNG CƠ CHẾ đã từng làm hỏng Sheep ở v4/v5. Rút chu kỳ xuống
    // còn 0,2s (~5 chu kỳ trọn vẹn/cửa sổ) để Duty_cycle đo trong từng cửa
    // sổ ổn định quanh giá trị thiết kế, giữ nguyên tỉ lệ ON/OFF (~0,15)
    // nên bản chất "lưu lượng thưa, ngắt quãng nhiều" của Mice không đổi.
    auto installMiceFlow = [&](int clientNode, int serverNode, uint16_t port)
    {
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                               InetSocketAddress(Ipv4Address::GetAny(), port));
        sink.Install(nodes.Get(serverNode));

        OnOffHelper mice("ns3::UdpSocketFactory",
                          InetSocketAddress(ifaceOfServerNode(serverNode), port));
        uint32_t rateKbps = static_cast<uint32_t>(miceRateRv->GetValue());
        mice.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string(rateKbps) + "kbps")));
        mice.SetAttribute("PacketSize", UintegerValue(150));   // dưới ngưỡng Mice 372B
        mice.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean=0.03]"));
        mice.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.17]"));

        auto app = mice.Install(nodes.Get(clientNode));
        app.Start(Seconds(1.0 + startJitter->GetValue()));
        app.Stop(Seconds(simTime));
    };

    // ---- Cài đặt 300 luồng: mỗi vòng lặp i tạo đồng thời 1 luồng Elephant +
    // 1 luồng Sheep + 1 luồng Mice, dùng cùng cặp (clientNode, serverNode)
    // xoay vòng nhưng KHÁC cổng (base port riêng theo từng lớp + i), nên
    // không xung đột với nhau dù chung node.
    uint16_t elephantBasePort = 20000;
    uint16_t sheepBasePort    = 21000;
    uint16_t miceBasePort     = 22000;

    for (int i = 0; i < NUM_FLOWS_PER_CLASS; i++)
    {
        int clientNode = i % NUM_CLIENT_NODES;          // 0..9,  xoay vòng
        int serverNode = 12 + (i % NUM_SERVER_NODES);    // 12..49, xoay vòng

        installElephantFlow(clientNode, serverNode, elephantBasePort + i);
        installSheepFlow(clientNode, serverNode, sheepBasePort + i);
        installMiceFlow(clientNode, serverNode, miceBasePort + i);
    }

    // ================= CSV (1 FILE DUY NHẤT) =================
    // Tên file gắn theo qdisc/seed/run để mỗi tổ hợp (thuật toán x lần lặp)
    // sinh ra 1 file riêng, không ghi đè lẫn nhau khi so sánh nhiều thuật toán.
    //
    // Schema hợp nhất — mỗi event_type chỉ điền các cột liên quan, các cột
    // còn lại để trống (không phải NaN giả, mà là "không áp dụng cho loại
    // sự kiện này"):
    //   TX / RX          : điền đủ protocol..tcp_seq, kèm rtt_ms snapshot
    //   ENQUEUE/DEQUEUE/DROP : điền direction, src_ip, dst_ip, identification,
    //                          queue_size, d_queue_ms (chỉ DEQUEUE), rtt_ms
    //   RTT_UPDATE        : điền rtt_context, rtt_ms
    // MỚI (v4): thêm cột tcp_seq (số thứ tự TCP) ngay sau tcp_ack_flag, phục
    // vụ tính Retransmission Rate (công thức 4.4) — trước đây không có cột
    // này nên không thể phát hiện gói TCP truyền lại ở bước tiền xử lý.
    // MỚI (v7): thêm cột src_port/dst_port ngay sau tcp_seq — trước đây
    // KHÔNG có cột này khiến Python định danh luồng chỉ bằng (src_ip,
    // dst_ip, protocol), gộp nhầm nhiều luồng khác lớp (Mice/Sheep/Elephant)
    // dùng chung 1 cặp IP thành 1 "luồng" duy nhất (xem chú thích chi tiết
    // tại TraceIpHeaderCommon).
    std::string baseTag = qdisc + "_seed" + std::to_string(seed) + "_run" + std::to_string(run);
    std::string csvFileName = "out_ip_header_" + baseTag + ".csv";

    csv.open(csvFileName);
    csv << "Timestamp,event_type,direction,src_ip,dst_ip,protocol,tos,ttl,"
           "identification,fragment_offset,MF,DF,payload_size,tcp_ack_flag,tcp_seq,"
           "src_port,dst_port,"
           "queue_size,d_queue_ms,rtt_context,rtt_ms"
        << std::endl;

    Config::Connect(
        "/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
        MakeCallback(&TraceIpHeaderTx));

    Config::Connect(
        "/NodeList/*/$ns3::Ipv4L3Protocol/Rx",
        MakeCallback(&TraceIpHeaderRx));

    // RTT thật lấy từ bộ ước lượng RTT của TCP tại tầng socket (mọi node, mọi socket).
    // QUAN TRỌNG: KHÔNG được gọi Config::Connect cho đường dẫn này ngay tại t=0,
    // vì lúc đó SocketList của TcpL4Protocol vẫn còn RỖNG (socket TCP chỉ được
    // tạo ra khi ứng dụng thực sự Start()). Gọi khi danh sách rỗng sẽ khiến ns-3
    // báo lỗi NS_FATAL "Could not connect callback" và dừng hẳn chương trình.
    // => Hoãn lại bằng Simulator::Schedule, chạy SAU thời điểm mọi ứng dụng đã
    // Start(Seconds(1.0)), để lúc đó socket đã tồn tại và việc kết nối trace
    // chắc chắn thành công.
    Simulator::Schedule(Seconds(1.05), [](){
        Config::Connect(
            "/NodeList/*/$ns3::TcpL4Protocol/SocketList/*/RTT",
            MakeCallback(&RttTraceHandler));
    });

    // ================= NETANIM =================
    AnimationInterface anim("simulation_la50_v2.xml");
    anim.EnablePacketMetadata(true);

    for (int i = 0; i < 10; i++)
        anim.SetConstantPosition(nodes.Get(i), 0, 100 - i * 10);

    anim.SetConstantPosition(nodes.Get(10), 50, 50);
    anim.SetConstantPosition(nodes.Get(11), 150, 50);

    for (int i = 12; i < 50; i++)
        anim.SetConstantPosition(nodes.Get(i), 200, 100 - (i - 12) * 3);

    anim.UpdateNodeDescription(nodes.Get(10), "Router-1 (N11)");
    anim.UpdateNodeDescription(nodes.Get(11), "Router-2 (N12)");

    anim.SetMaxPktsPerTraceFile(100000);

    // ================= RUN =================
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    Simulator::Destroy();
    csv.close();

    std::cout << "Simulation completed! (qdisc=" << qdisc
               << ", seed=" << seed << ", run=" << run << ")\n";
    std::cout << "- CSV (1 file duy nhat, gop TX/RX/ENQUEUE/DEQUEUE/DROP/RTT_UPDATE): "
              << csvFileName << "\n";
    std::cout << "- Animation: simulation_la50_v2.xml\n";
    std::cout << "- Bottleneck queue: 300p, qdisc=" << qdisc << "\n";
    std::cout << "- TCP window: SndBufSize=RcvBufSize=16384 bytes (16KB)\n";
    std::cout << "- Tong so luong: " << (NUM_FLOWS_PER_CLASS * 3)
              << " (client N1-N10, server N13-N50, xoay vong, deu qua bottleneck)\n";
    std::cout << "  + Elephant: " << NUM_FLOWS_PER_CLASS
              << " luong TCP BulkSend lien tuc, SendSize=1200B\n";
    std::cout << "  + Sheep   : " << NUM_FLOWS_PER_CLASS
              << " luong UDP OnOff, 30-80kbps khi ON, PacketSize=600B, duty~0.77\n";
    std::cout << "  + Mice    : " << NUM_FLOWS_PER_CLASS
              << " luong UDP OnOff, 40-120kbps khi ON, PacketSize=150B, duty~0.15\n";

    return 0;
}
