import cv2
import time
import os
from ultralytics import YOLO

# ================================================================
# 1. NẠP MÔ HÌNH
# ================================================================
print("🔄 Đang nạp mô hình nhận diện ổ gà...")
model = YOLO("python_ai/weights/best.pt")
print("✅ Mô hình đã sẵn sàng!")

# ================================================================
# 2. CẤU HÌNH THƯ MỤC CHỨA NHIỀU VIDEO
# ================================================================
# Đường dẫn dẫn thẳng vào thư mục python_ai
video_folder = os.path.abspath("python_ai")

# Lấy tất cả các file video có đuôi .mp4 trong thư mục python_ai
# (Ngoại trừ các file hệ thống ẩn nếu có)
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

if not video_files:
    print(f"❌ Không tìm thấy file video .mp4 nào trong thư mục: {video_folder}")
    print("💡 Hãy copy thêm các file video hành trình của bạn vào mục này nhé!")
    exit()

print(f"📚 Tìm thấy {len(video_files)} video để xử lý: {video_files}")

# ================================================================
# 3. VÒNG LẶP DUYỆT QUA TỪNG FILE VIDEO
# ================================================================
for video_name in video_files:
    video_source = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở file video: {video_source}")
        continue # Nếu lỗi 1 file thì bỏ qua, chạy tiếp file sau

    print(f"\n🚀 ĐANG XỬ LÝ VIDEO: {video_name}")
    print("👉 Ấn 'q' trên bàn phím để BỎ QUA video này (hoặc chuyển tiếp).\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"🏁 Video {video_name} đã kết thúc.")
            break

        # --- FIX 1: Gán annotated_frame mặc định trước khi inference ---
        annotated_frame = frame.copy()

        # --- FIX 2: Đo FPS chỉ bao quanh inference ---
        t0 = time.perf_counter()

        # FIX 3: Chạy AI tối ưu kích thước và độ tự tin conf=0.25
        results = model(frame, stream=True, imgsz=320, conf=0.25)  

        for r in results:
            annotated_frame = r.plot()

        fps = 1 / (time.perf_counter() - t0)

        # --- Hiển thị Tên File và FPS lên màn hình ---
        cv2.putText(
            annotated_frame,
            f"File: {video_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("YOLOv8 Dashcam - Multi-Video Mode", annotated_frame)

        # Bắt sự kiện phím 'q' để nhảy sang video kế tiếp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"⏭️ Người dùng chủ động bỏ qua video: {video_name}")
            break

    # Giải phóng video hiện tại để chuẩn bị nạp video mới
    cap.release()

# ================================================================
# 4. GIẢI PHÓNG TOÀN BỘ TÀI NGUYÊN
# ================================================================
cv2.destroyAllWindows()
print("\n👋 Đã đóng an toàn và hoàn thành toàn bộ danh sách video!")