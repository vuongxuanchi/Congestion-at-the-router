Thành phần	Thiết lập
Số nút	50 (N1–N50; trong mã đánh số từ 0)
Liên kết truy cập	20 Mbps, trễ 5 ms; N1–N10 nối N11, N13–N50 nối N12
Liên kết thắt cổ chai	N11–N12, 10 Mbps, trễ 50 ms
Hàng đợi tại thắt cổ chai	300 gói; chọn bằng --qdisc: droptail (mặc định), codel, fqcodel, pie, cake, red, ared
TCP	bộ đệm gửi và nhận 16 KB; kích thước đoạn tối đa (MSS) 1400 byte
Thời gian mô phỏng	180 s
Sinh ngẫu nhiên	--seed (mặc định 10), --run (mặc định 1)
