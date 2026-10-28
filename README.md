
<table>
    <thead> MÔ HÌNH THỰC NGHIỆM
        <tr>
            <th>Thành phần</th>
            <th>Thiết lập</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Số nút</td>
            <td>50 (N1–N50; trong mã đánh số từ 0)</td>
        </tr>
        <tr>
            <td>Liên kết truy cập</td>
            <td>20 Mbps, trễ 5 ms; N1–N10 nối N11, N13–N50 nối N12</td>
        </tr>
        <tr>
            <td>Liên kết thắt cổ chai</td>
            <td>N11–N12, 10 Mbps, trễ 50 ms</td>
        </tr>
        <tr>
            <td>Hàng đợi tại thắt cổ chai</td>
            <td>
                300 gói; chọn bằng --qdisc:
                droptail (mặc định), codel, fqcodel, pie,
                cake, red, ared
            </td>
        </tr>
        <tr>
            <td>TCP</td>
            <td>
                Bộ đệm gửi và nhận 16 KB;
                kích thước đoạn tối đa (MSS) 1400 byte
            </td>
        </tr>
        <tr>
            <td>Thời gian mô phỏng</td>
            <td>180 s</td>
        </tr>
        <tr>
            <td>Sinh ngẫu nhiên</td>
            <td>
                --seed (mặc định 10), --run (mặc định 1)
            </td>
        </tr>
    </tbody>
</table>

