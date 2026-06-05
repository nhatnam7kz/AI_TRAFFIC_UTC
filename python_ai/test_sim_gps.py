import cv2
import time
import csv
import os
import math
from datetime import datetime
from ultralytics import YOLO

# ================================================================
# 1. HAVERSINE — TÍNH KHOẢNG CÁCH THỰC TẾ GIỮA 2 TỌA ĐỘ GPS
# ================================================================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Bán kính Trái Đất (mét)
    rad_lat1, rad_lon1, rad_lat2, rad_lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rad_lat2 - rad_lat1
    dlon = rad_lon2 - rad_lon1
    a = math.sin(dlat/2)**2 + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# ================================================================
# 2. ĐỌC CẤU HÌNH GPS GIẢ LẬP TỪ FILE video2.txt
#    Định dạng: latitude,longitude,speed_kmh
# ================================================================
def load_gps_config(config_path="video2.txt"):
    current_lat, current_lon, speed = 21.025944, 105.801642, 40.0
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    current_lat = float(parts[0])
                    current_lon = float(parts[1])
                    speed       = float(parts[2])
                    break
    return current_lat, current_lon, speed

# ================================================================
# 3. KHỞI TẠO
# ================================================================
start_lat, start_lon, sim_speed = load_gps_config()
current_lat = start_lat
current_lon = start_lon

print("🔄 Đang nạp mô hình...")
model = YOLO("python_ai/weights/best.pt")
print("✅ Mô hình đã sẵn sàng!")

video_path = "python_ai/video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Không mở được video tại: {video_path}")
    exit()

# Tạo thư mục log
session_id = "TEST_SIM_" + datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir    = os.path.join("python_ai/logs", session_id)
os.makedirs(log_dir, exist_ok=True)
log_file   = os.path.join(log_dir, "pothole_log.csv")

print(f"📁 Session folder: {log_dir}")
print(f"\n🚀 ĐANG CHẠY GIẢ LẬP XE CHẠY THẲNG ({sim_speed} km/h)... Ấn 'q' để THOÁT.\n")

# ================================================================
# 4. VÒNG LẶP CHÍNH
# ================================================================
frame_count     = 0
last_log_time   = 0.0
last_logged_lat = 0.0
last_logged_lon = 0.0
t_last_frame    = time.perf_counter()

# FIX #6: Mở file CSV một lần duy nhất, giữ handle suốt session
with open(log_file, "w", newline="", encoding="utf-8") as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(["timestamp", "latitude", "longitude", "speed_kmh", "confidence", "frame"])

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("🎬 Hết video hành trình.")
                break

            frame_count += 1

            # FIX #3: Đo dt toàn pipeline → FPS thực
            t_now        = time.perf_counter()
            dt           = t_now - t_last_frame
            t_last_frame = t_now
            fps          = 1.0 / dt if dt > 0 else 0.0

            # --- GIẢ LẬP XE CHẠY THẲNG LÊN PHÍA BẮC ---
            distance_moved = (sim_speed / 3.6) * dt    # mét
            current_lat   += distance_moved / 111111.0 # độ vĩ
            # lon không thay đổi (xe chạy thẳng Bắc)
            # --------------------------------------------

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # --- AI NHẬN DIỆN ---
            results = model(frame, stream=True, imgsz=320, conf=0.25)

            pothole_detected = False
            max_conf         = 0.0
            annotated_frame  = frame.copy()

            for r in results:
                annotated_frame = r.plot()
                if r.boxes is not None and len(r.boxes) > 0:
                    pothole_detected = True
                    max_conf = float(r.boxes.conf.max())

            # FIX #1: Log khi ĐÃ QUA 1 giây VÀ (lần đầu HOẶC cách điểm cũ >= 3m)
            if pothole_detected:
                dist        = calculate_distance(current_lat, current_lon,
                                                 last_logged_lat, last_logged_lon)
                is_first    = (last_logged_lat == 0.0 and last_logged_lon == 0.0)
                time_ok     = (time.time() - last_log_time) > 1.0
                distance_ok = (dist >= 3.0 or is_first)

                if time_ok and distance_ok:
                    last_log_time   = time.time()
                    last_logged_lat = current_lat
                    last_logged_lon = current_lon

                    writer.writerow([
                        timestamp_str,
                        round(current_lat, 6),
                        round(current_lon, 6),
                        sim_speed,
                        round(max_conf, 3),
                        frame_count
                    ])
                    csv_f.flush()  # Ghi ngay xuống disk, không đợi close

                    print(f"📍 Ổ gà [GIẢ LẬP] tại: {current_lat:.6f}, {current_lon:.6f} "
                          f"| Cách điểm cũ: {dist:.1f}m | Conf: {max_conf:.3f}")

            # --- UI OVERLAY ---
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (420, 115), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

            cv2.putText(annotated_frame, f"FPS: {fps:.1f} (SIM MODE)",
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 255, 0),   2)
            cv2.putText(annotated_frame, f"LAT: {current_lat:.6f}  LON: {current_lon:.6f}",
                        (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"Speed: {sim_speed} km/h | Moved: {distance_moved:.3f}m",
                        (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            if pothole_detected:
                cv2.putText(annotated_frame, "!!! POTHOLE DETECTED",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("YOLOv8 Pothole - GPS Simulation Test", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Người dùng dừng.")
                break

    finally:
        # FIX #2: Luôn giải phóng tài nguyên dù có crash hay không
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n👋 Đã đóng an toàn.")
        print(f"📊 Tổng frames xử lý: {frame_count}")
        print(f"📁 Log đã lưu tại:     {log_file}")

# ================================================================
# 5. TẠO BẢN ĐỒ HTML SAU KHI KẾT THÚC
# ================================================================
def generate_map(log_file, log_dir, session_id):
    """Đọc CSV và xuất file map.html dùng Leaflet + Google Maps tiles."""
    points = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(row)
    except Exception as e:
        print(f"⚠ Không đọc được file log: {e}")
        return

    if not points:
        print("⚠ Không có dữ liệu ổ gà để vẽ bản đồ.")
        return

    center_lat = sum(float(p["latitude"])  for p in points) / len(points)
    center_lon = sum(float(p["longitude"]) for p in points) / len(points)

    markers_js = ""
    for p in points:
        try:
            conf  = float(p["confidence"])
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
        except (KeyError, ValueError) as e:
            print(f"⚠ Bỏ qua điểm lỗi: {e}")
            continue

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
            font-size: 14px;
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
    try:
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"🗺  Bản đồ đã lưu tại: {html_file}")
    except Exception as e:
        print(f"⚠ Không ghi được file map: {e}")

generate_map(log_file, log_dir, session_id)