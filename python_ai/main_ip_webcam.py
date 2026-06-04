import cv2
import time
import csv
import os
import requests
from threading import Thread, Lock
from datetime import datetime
from ultralytics import YOLO

# ================================================================
# 1. NẠP MÔ HÌNH
# ================================================================
print("🔄 [IP WEBCAM + GPS MODE] Đang nạp mô hình...")
model = YOLO("python_ai/weights/best.pt")
print("✅ Mô hình đã sẵn sàng!")

# ================================================================
# 2. GPS THẬT — đọc từ Termux server trên phone
# ================================================================
class HTTPLocationGPS:
    def __init__(self, host, port=8081):
        self.url = f"http://{host}:{port}/gps"
        self.lat = 21.027763
        self.lon = 105.834160
        self.speed = 0.0
        self.lock = Lock()

    def start(self):
        Thread(target=self._poll_loop, daemon=True).start()
        return self

    def _poll_loop(self):
        while True:
            try:
                r = requests.get(self.url, timeout=3)
                data = r.json()
                with self.lock:
                    self.lat = data["lat"]
                    self.lon = data["lon"]
                    self.speed = data["speed"]
            except:
                pass
            time.sleep(1)

    def get(self):
        with self.lock:
            return {
                "lat": round(self.lat, 6),
                "lon": round(self.lon, 6),
                "speed": round(self.speed, 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

gps = HTTPLocationGPS("192.168.1.90", port=8081).start()
print("✅ GPS đã kết nối!")

# ================================================================
# 3. TẠO FOLDER SESSION + LOG FILE
# ================================================================
session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join("python_ai/logs", session_id)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pothole_log.csv")

with open(log_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "latitude", "longitude", "speed_kmh", "confidence", "frame"])

print(f"📁 Session folder: {log_dir}")

frame_count = 0
last_log_time = 0

# ================================================================
# 4. IP WEBCAM CLASS — chống delay
# ================================================================
ip_address = "http://192.168.1.90:8080"
video_source = f"{ip_address}/video"

class IPWebcamCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Không kết nối được: {src}")
        self.lock = Lock()
        self.success = False
        self.frame = None
        self.running = True

    def start(self):
        Thread(target=self.update, daemon=True).start()
        timeout = time.time() + 5.0
        while self.frame is None:
            if time.time() > timeout:
                raise RuntimeError("❌ Không nhận được frame sau 5 giây.")
            time.sleep(0.05)
        return self

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.cap.grab()
                success, frame = self.cap.retrieve()
                with self.lock:
                    self.success = success
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.success, self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()

# ================================================================
# 5. KHỞI ĐỘNG
# ================================================================
print(f"🔗 Đang kết nối tới: {video_source} ...")
try:
    cap = IPWebcamCapture(video_source).start()
except RuntimeError as e:
    print(e)
    exit()

print("\n🚀 ĐANG CHẠY REAL-TIME + GPS THẬT... Ấn 'q' để THOÁT.\n")

# ================================================================
# 6. VÒNG LẶP CHÍNH
# ================================================================
while True:
    success, frame = cap.read()
    if not success or frame is None:
        print("❌ Mất tín hiệu.")
        break

    frame_count += 1
    annotated_frame = frame.copy()
    gps_data = gps.get()

    # Inference
    t0 = time.perf_counter()
    results = model(frame, stream=True, imgsz=320, conf=0.25)

    pothole_detected = False
    max_conf = 0.0

    for r in results:
        annotated_frame = r.plot()
        if r.boxes is not None and len(r.boxes) > 0:
            pothole_detected = True
            max_conf = float(r.boxes.conf.max())

    fps = 1 / (time.perf_counter() - t0)

    # Log khi phát hiện ổ gà — tối đa 1 lần/giây
    if pothole_detected and (time.time() - last_log_time) > 1.0:
        last_log_time = time.time()
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                gps_data["timestamp"],
                gps_data["lat"],
                gps_data["lon"],
                gps_data["speed"],
                round(max_conf, 3),
                frame_count
            ])
        print(f"📍 Ổ gà tại: {gps_data['lat']}, {gps_data['lon']} | conf={max_conf:.2f}")

    # Hiển thị thông tin lên video
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (380, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame,
                f"LAT: {gps_data['lat']}  LON: {gps_data['lon']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.putText(annotated_frame,
                f"Speed: {gps_data['speed']} km/h",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    if pothole_detected:
        cv2.putText(annotated_frame,
                    "⚠ POTHOLE DETECTED",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Pothole + GPS", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================================================
# 7. TẠO BẢN ĐỒ HTML SAU KHI KẾT THÚC
# ================================================================
def generate_map(log_file, log_dir, session_id):
    points = []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(row)

    if not points:
        print("⚠ Không có dữ liệu ổ gà để vẽ bản đồ.")
        return

    center_lat = sum(float(p["latitude"]) for p in points) / len(points)
    center_lon = sum(float(p["longitude"]) for p in points) / len(points)

    markers_js = ""
    for p in points:
        conf = float(p["confidence"])
        color = "green" if conf >= 0.6 else "orange" if conf >= 0.4 else "red"
        markers_js += f"""
        L.circleMarker([{p["latitude"]}, {p["longitude"]}], {{
            color: '{color}', fillColor: '{color}',
            fillOpacity: 0.8, radius: 8
        }}).addTo(map).bindPopup(
            "<b>⚠ Ổ gà</b><br>" +
            "🕐 {p['timestamp']}<br>" +
            "📍 {p['latitude']}, {p['longitude']}<br>" +
            "🚗 {p['speed_kmh']} km/h<br>" +
            "🎯 Conf: {p['confidence']}"
        );
        """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bản đồ ổ gà — {session_id}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; }}
        #info {{
            position: absolute; top: 10px; left: 50px;
            z-index: 1000; background: white;
            padding: 10px 16px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
    </style>
</head>
<body>
<div id="info">
    🕐 {points[0]['timestamp']} → {points[-1]['timestamp']} &nbsp;|&nbsp;
    📍 Tổng ổ gà: <b>{len(points)}</b> &nbsp;|&nbsp;
    🟢 Conf≥0.6 &nbsp; 🟠 0.4–0.6 &nbsp; 🔴 &lt;0.4
</div>
<div id="map"></div>
<script>
    var map = L.map('map').setView([{center_lat}, {center_lon}], 16);
    L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{
        attribution: '© Google Maps'
    }}).addTo(map);
    {markers_js}
</script>
</body>
</html>"""

    html_file = os.path.join(log_dir, "map.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🗺 Bản đồ đã lưu tại: {html_file}")

# ================================================================
# 8. KẾT THÚC
# ================================================================
cap.stop()
cv2.destroyAllWindows()
print(f"\n👋 Đã đóng an toàn.")
print(f"📊 Tổng frames xử lý: {frame_count}")
print(f"📁 Log đã lưu tại: {log_file}")

generate_map(log_file, log_dir, session_id)