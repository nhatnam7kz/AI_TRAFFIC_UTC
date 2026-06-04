import cv2
import time
from threading import Thread, Lock
from ultralytics import YOLO

# ================================================================
# 1. NẠP MÔ HÌNH
# ================================================================
print("🔄 [REAL-TIME MODE] Đang nạp mô hình nhận diện ổ gà...")
model = YOLO("python_ai/weights/best.pt")
print("✅ Mô hình đã sẵn sàng!")

CAMERA_INDEX = 0

# ================================================================
# 2. CLASS ĐỌC CAMERA MULTI-THREAD
# ================================================================
class RealTimeVideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Không kết nối được camera index {src}")

        self.lock = Lock()          # FIX 1: Lock bảo vệ self.frame khỏi race condition
        self.success = False
        self.frame = None
        self.running = True

    def start(self):
        Thread(target=self.update, daemon=True).start()
        # FIX 2: Chờ thread đọc được frame đầu tiên trước khi trả về
        timeout = time.time() + 3.0  # tối đa chờ 3 giây
        while self.frame is None:
            if time.time() > timeout:
                raise RuntimeError("❌ Camera không trả về frame sau 3 giây.")
            time.sleep(0.05)
        return self

    def update(self):
        while self.running:
            success, frame = self.cap.read()
            # FIX 3: Bỏ sleep cố định — cap.read() tự block theo FPS camera
            with self.lock:
                self.success = success
                self.frame = frame

    def read(self):
        # FIX 1: Đọc frame qua lock để tránh đọc frame đang được ghi
        with self.lock:
            return self.success, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        time.sleep(0.1)  # Chờ thread update kết thúc vòng hiện tại
        self.cap.release()

# ================================================================
# 3. KHỞI ĐỘNG CAMERA
# ================================================================
try:
    cap = RealTimeVideoCapture(CAMERA_INDEX).start()
except RuntimeError as e:
    print(e)
    exit()

print("\n🚀 CAMERA THỜI GIAN THỰC ĐANG CHẠY... Ấn 'q' để THOÁT.\n")

# ================================================================
# 4. VÒNG LẶP CHÍNH
# ================================================================
while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("⚠️ Mất tín hiệu camera.")
        break

    # Gán mặc định tránh crash khi không có detection
    annotated_frame = frame.copy()

    # Đo FPS chỉ bao quanh inference
    t0 = time.perf_counter()

    results = model(frame, stream=True, imgsz=320, conf=0.25)

    for r in results:
        annotated_frame = r.plot()

    fps = 1 / (time.perf_counter() - t0)

    cv2.putText(
        annotated_frame,
        f"Real-time FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2
    )

    cv2.imshow("YOLOv8 Pothole - Real-time Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================================================
# 5. GIẢI PHÓNG TÀI NGUYÊN
# ================================================================
cap.stop()
cv2.destroyAllWindows()
print("👋 Đã đóng luồng camera an toàn.")