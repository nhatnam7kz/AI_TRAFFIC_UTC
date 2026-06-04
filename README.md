# 🚗 Hệ Thống Phát Hiện Ổ Gà Thời Gian Thực - YOLOv8 (Pothole Detection)

Dự án nghiên cứu và phát triển hệ thống nhận diện ô vuông vị trí ổ gà (`Pothole`) dựa trên luồng dữ liệu hình ảnh hành trình thực tế. Hệ thống sử dụng mô hình tối ưu **YOLOv8** nhằm phục vụ bài toán số hóa hạ tầng giao thông và đưa ra cảnh báo sớm, tăng tính an toàn cho người tham gia giao thông.

---

## ✨ Các Tính Năng Nổi Bật (Features)
* **Xử lý Đa luồng (Multi-threading):** Triệt tiêu hoàn toàn hiện tượng trễ hình (delay) khi xử lý luồng camera thực tế.
* **Chống Crash Hệ Thống [FIX 1]:** Đảm bảo chương trình chạy liên tục, mượt mà ngay cả khi khung hình không xuất hiện ổ gà nào.
* **Cách Ly Phép Đo Hiệu Năng [FIX 2]:** Đo lường chỉ số FPS chuẩn xác, phản ánh đúng tốc độ xử lý phần cứng của mô hình (Inference Time).
* **Tối Ưu Hóa Tài Nguyên [FIX 3]:** Ép kích thước ảnh đầu vào về `imgsz=320` và ngưỡng tin cậy `conf=0.25`, giúp chạy tốt trên các thiết bị cấu hình yếu, hạn chế tối đa sinh nhiệt.
* **Hỗ trợ Đa Nguồn Vào (Multi-Input):**
  * Quét tự động danh sách nhiều file video liên tục.
  * Nhận diện qua Webcam/Camera hành trình kết nối trực tiếp.
  * Stream luồng xử lý không trễ từ camera điện thoại thông qua **IP Webcam**.

---

## 📂 Cấu Trúc Thư Mục Dự Án (Project Structure)
```text
YOLO_Dashcam/
│
├── python_ai/
│   ├── weights/
│   │   └── best.pt             # Bộ não AI (Trọng số mô hình sau khi Train)
│   │
│   ├── main_video.py           # Chạy phân tích tuần tự trên 1 video mẫu
│   ├── main_realtime.py        # Chạy Real-time qua Webcam (Multi-threading)
│   ├── main_multi_videos.py    # Tự động quét và hiển thị hàng loạt video (.mp4)
│   └── main_ip_webcam.py       # Kết nối luồng không trễ với app IP Webcam điện thoại
│
├── .gitignore                  # Cấu hình bộ lọc loại trừ các file video nặng/file rác
├── requirements.txt            # Danh sách các thư viện cần thiết để hệ thống hoạt động
└── README.md                   # Hướng dẫn sử dụng dự án (File này)


pip install -r requirements.txt
