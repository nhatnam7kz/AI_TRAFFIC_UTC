"""
Pothole Detection — Real-time YOLO + IP Webcam + GPS
=====================================================
Bản cải tiến v2 — fix delay camera:
  - Thay MJPEG stream → JPEG snapshot (/shot.jpg) để loại bỏ buffer lag
  - Fallback tự động về MJPEG stream nếu snapshot thất bại
  - CAM_THREAD_SLEEP tăng lên 0.016s (60Hz) giảm CPU spin vô ích
  - Thêm CAM_SNAPSHOT_MODE config để dễ bật/tắt
  - emit_json_event dùng append-only (JSONL) thay vì đọc-ghi toàn bộ file
  - GPS stale detection: cảnh báo nếu GPS không cập nhật quá N giây
  - Thêm GPS_STALE_SEC config
  - Wrap cv2.imshow trong try/except để tránh crash headless
  - Các fix từ bản trước giữ nguyên
"""

import cv2
import time
import csv
import json
import os
import math
import urllib.request
import numpy as np
import requests
from threading import Thread, Lock
from datetime import datetime, timezone
from ultralytics import YOLO

# ================================================================
# CONFIG
# ================================================================
PHONE_IP        = "10.50.91.154"
GPS_PORT        = 8081
CAM_PORT        = 8080
YOLO_WEIGHTS    = "python_ai/weights/best.pt"
YOLO_IMGSZ      = 320
YOLO_CONF       = 0.25
LOG_DIR_BASE    = "python_ai/logs"

# Device ID — Backend dùng để đồng thuận
DEVICE_ID       = "cam_001"

# Chống log trùng lặp
MIN_LOG_INTERVAL_SEC  = 1.0
MIN_LOG_DISTANCE_M    = 3.0

# Retry camera
MAX_CONSECUTIVE_FAILURES = 30
CAM_RETRY_SLEEP          = 0.03

# ── FIX DELAY ──────────────────────────────────────────────────
# True  → dùng /shot.jpg (snapshot, KHÔNG có buffer, delay ~0)
# False → fallback về /video (MJPEG stream, có buffer lag)
CAM_SNAPSHOT_MODE   = True
CAM_SNAPSHOT_FPS    = 30                        # pull tối đa N frame/giây
CAM_THREAD_SLEEP    = 1.0 / CAM_SNAPSHOT_FPS   # ≈ 0.033s, không spin vô ích
CAM_SNAPSHOT_TIMEOUT = 2.0                      # timeout mỗi request snapshot (giây)

# GPS stale detection
GPS_STALE_SEC = 5.0   # cảnh báo nếu GPS không cập nhật quá 5 giây


# ================================================================
# HAVERSINE
# ================================================================
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ================================================================
# GENERATE MAP (giữ nguyên, chỉ đọc CSV)
# ================================================================
def generate_map(log_file_path: str, dir_path: str, sess_id: str) -> None:
    points = []
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"timestamp", "latitude", "longitude", "speed_kmh", "confidence"}
            for row in reader:
                if not required.issubset(row.keys()):
                    continue
                try:
                    float(row["latitude"])
                    float(row["longitude"])
                    float(row["confidence"])
                except ValueError:
                    continue
                points.append(row)
    except Exception as e:
        print(f"⚠ Không đọc được file log: {e}")
        return

    if not points:
        print("⚠ Không có dữ liệu ổ gà trong phiên này. Bỏ qua xuất bản đồ.")
        return

    center_lat = sum(float(p["latitude"])  for p in points) / len(points)
    center_lon = sum(float(p["longitude"]) for p in points) / len(points)

    markers_js = ""
    for p in points:
        conf  = float(p["confidence"])
        color = "green" if conf >= 0.6 else "orange" if conf >= 0.4 else "red"
        ts    = p["timestamp"].replace("'", "&#39;")
        lat   = p["latitude"]
        lon   = p["longitude"]
        spd   = p["speed_kmh"]
        markers_js += f"""
        L.circleMarker([{lat}, {lon}], {{
            color: '{color}', fillColor: '{color}',
            fillOpacity: 0.8, radius: 8
        }}).addTo(map).bindPopup(
            "<b>&#9888; &#7893; g&#224;</b><br>" +
            "&#128336; {ts}<br>" +
            "&#128205; {lat}, {lon}<br>" +
            "&#128663; {spd} km/h<br>" +
            "&#127919; Conf: {conf:.3f}"
        );
        """

    time_range = f"{points[0]['timestamp']} → {points[-1]['timestamp']}"
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Bản đồ ổ gà — {sess_id}</title>
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
    &#128336; {time_range} &nbsp;|&nbsp;
    &#128205; T&#7893;ng &#7893; g&#224;: <b>{len(points)}</b> &nbsp;|&nbsp;
    &#127abe; Conf&#8805;0.6 &nbsp; &#127811; 0.4&#8211;0.6 &nbsp; &#128308; &lt;0.4
</div>
<div id="map"></div>
<script>
    var map = L.map('map').setView([{center_lat}, {center_lon}], 16);
    L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{
        attribution: '&copy; Google Maps'
    }}).addTo(map);
    {markers_js}
</script>
</body>
</html>"""

    html_file = os.path.join(dir_path, "map.html")
    try:
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"🗺  Bản đồ đã lưu tại: {html_file}")
    except Exception as e:
        print(f"⚠ Không ghi được file map: {e}")


# ================================================================
# GPS — HTTP polling + stale detection
# ================================================================
class HTTPLocationGPS:
    """Poll GPS data từ endpoint JSON trên phone."""

    def __init__(self, host: str, port: int = 8081) -> None:
        self.url          = f"http://{host}:{port}/gps"
        self._lat         = 21.027763
        self._lon         = 105.834160
        self._spd         = 0.0
        self.ready        = False
        self._lock        = Lock()
        self._last_update = 0.0   # monotonic time của lần cập nhật cuối

    def start(self) -> "HTTPLocationGPS":
        Thread(target=self._poll_loop, daemon=True, name="gps-poller").start()
        return self

    def _poll_loop(self) -> None:
        while True:
            try:
                r    = requests.get(self.url, timeout=3)
                r.raise_for_status()
                data = r.json()
                with self._lock:
                    self._lat         = float(data["lat"])
                    self._lon         = float(data["lon"])
                    self._spd         = float(data.get("speed", 0.0))
                    self.ready        = True
                    self._last_update = time.monotonic()
            except Exception as e:
                print(f"⚠ GPS poll lỗi: {e}")
            time.sleep(1.0)

    def get(self) -> dict:
        """Trả về snapshot GPS tại thời điểm gọi (thread-safe)."""
        with self._lock:
            stale = (
                self.ready
                and (time.monotonic() - self._last_update) > GPS_STALE_SEC
            )
            return {
                "lat":             round(self._lat, 6),
                "lon":             round(self._lon, 6),
                "speed":           round(self._spd, 1),
                "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stale":           stale,   # True nếu GPS không cập nhật quá GPS_STALE_SEC
            }


# ================================================================
# JSON EMITTER — append-only JSONL (không đọc lại toàn bộ file)
# ================================================================
def emit_json_event(
    json_file: str,
    gps_data:  dict,
    confidence: float,
    frame_no:   int,
) -> None:
    """
    Ghi 1 event ổ gà vào file JSONL (1 JSON object mỗi dòng).
    Nhanh hơn nhiều so với load-append-dump toàn bộ mảng.
    Khi cần đọc lại: [json.loads(l) for l in open(file)]
    """
    event = {
        "lat":        gps_data["lat"],
        "lng":        gps_data["lon"],
        "type":       "pothole",
        "confidence": round(confidence, 3),
        "timestamp":  gps_data["timestamp"],
        "device_id":  DEVICE_ID,
        "speed_kmh":  gps_data["speed"],
        "frame":      frame_no,
    }
    try:
        with open(json_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠ Không ghi được JSON: {e}")


# ================================================================
# IP WEBCAM — snapshot mode (fix delay) + MJPEG fallback
# ================================================================
class IPWebcamCapture:
    """
    Snapshot mode (mặc định, CAM_SNAPSHOT_MODE=True):
        Pull /shot.jpg mỗi vòng lặp → luôn lấy frame MỚI NHẤT,
        không có buffer tích lũy, delay gần bằng 0.

    MJPEG fallback (CAM_SNAPSHOT_MODE=False):
        Dùng /video stream như cũ, grab()+retrieve() background thread.
    """

    def __init__(self, host: str, cam_port: int = 8080) -> None:
        self._host        = host
        self._cam_port    = cam_port
        self._lock        = Lock()
        self._frame       = None
        self._success     = False
        self._running     = True
        self._mode        = "snapshot" if CAM_SNAPSHOT_MODE else "mjpeg"

        if self._mode == "mjpeg":
            src = f"http://{host}:{cam_port}/video"
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if not self._cap.isOpened():
                raise RuntimeError(f"❌ Không kết nối được MJPEG: {src}")
        else:
            self._snapshot_url = f"http://{host}:{cam_port}/shot.jpg"
            self._cap          = None
            # Kiểm tra snapshot có hoạt động không
            try:
                with urllib.request.urlopen(
                    self._snapshot_url, timeout=CAM_SNAPSHOT_TIMEOUT
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError("Snapshot trả về status != 200")
                print(f"✅ Snapshot mode OK: {self._snapshot_url}")
            except Exception as e:
                print(f"⚠ Snapshot thất bại ({e}), chuyển sang MJPEG stream...")
                self._mode = "mjpeg"
                src = f"http://{host}:{cam_port}/video"
                self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self._cap.isOpened():
                    raise RuntimeError(f"❌ Cả snapshot lẫn MJPEG đều thất bại.")

    def start(self, timeout: float = 5.0) -> "IPWebcamCapture":
        target = self._update_snapshot if self._mode == "snapshot" else self._update_mjpeg
        Thread(target=target, daemon=True, name="cam-capture").start()
        deadline = time.monotonic() + timeout
        while self._frame is None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"❌ Không nhận được frame sau {timeout:.0f} giây.")
            time.sleep(0.05)
        print(f"✅ Camera sẵn sàng (mode: {self._mode})")
        return self

    # ── Snapshot thread ──────────────────────────────────────────
    def _update_snapshot(self) -> None:
        while self._running:
            try:
                with urllib.request.urlopen(
                    self._snapshot_url, timeout=CAM_SNAPSHOT_TIMEOUT
                ) as resp:
                    raw = resp.read()
                arr = np.frombuffer(raw, dtype=np.uint8)
                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                with self._lock:
                    self._success = frm is not None
                    self._frame   = frm
            except Exception as e:
                print(f"⚠ Snapshot lỗi: {e}")
                with self._lock:
                    self._success = False
            time.sleep(CAM_THREAD_SLEEP)

    # ── MJPEG thread (fallback) ───────────────────────────────────
    def _update_mjpeg(self) -> None:
        while self._running:
            if self._cap.isOpened():
                grabbed = self._cap.grab()
                if grabbed:
                    ok, frm = self._cap.retrieve()
                    with self._lock:
                        self._success = ok
                        self._frame   = frm
            time.sleep(CAM_THREAD_SLEEP)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._success, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        time.sleep(CAM_THREAD_SLEEP * 2)
        if self._cap is not None:
            self._cap.release()


# ================================================================
# MAIN
# ================================================================
def main() -> None:
    # --- 1. Nạp mô hình ---
    print("🔄 [IP WEBCAM + GPS MODE] Đang nạp mô hình...")
    model = YOLO(YOLO_WEIGHTS)
    print("✅ Mô hình đã sẵn sàng!")

    # --- 2. Khởi động GPS ---
    gps = HTTPLocationGPS(PHONE_IP, port=GPS_PORT).start()
    print("✅ GPS đã kết nối (đang chờ tín hiệu đầu tiên)...")

    # --- 3. Tạo folder session ---
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = os.path.join(LOG_DIR_BASE, session_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file   = os.path.join(log_dir, "pothole_log.csv")
    json_file  = os.path.join(log_dir, "pothole_events.jsonl")  # JSONL thay vì JSON
    print(f"📁 Session folder: {log_dir}")

    # --- 4. Kết nối camera ---
    print(f"🔗 Đang kết nối camera tới {PHONE_IP}:{CAM_PORT} ...")
    try:
        cap = IPWebcamCapture(PHONE_IP, CAM_PORT).start()
    except RuntimeError as e:
        print(e)
        return
    print("\n🚀 ĐANG CHẠY REAL-TIME + GPS THẬT... Ấn 'q' để THOÁT.\n")

    # --- 5. State ---
    frame_count      = 0
    retries          = 0
    last_log_time    = 0.0
    last_logged_pos: tuple[float, float] | None = None
    window_name      = "YOLOv8 Pothole + GPS"
    t_last           = time.perf_counter()

    with open(log_file, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["timestamp", "latitude", "longitude", "speed_kmh", "confidence", "frame"])

        try:
            while True:
                # --- FPS ---
                t_now  = time.perf_counter()
                dt     = t_now - t_last
                t_last = t_now
                fps    = 1.0 / dt if dt > 0 else 0.0

                # --- Đọc frame ---
                success, frame = cap.read()
                if not success or frame is None:
                    retries += 1
                    if retries > MAX_CONSECUTIVE_FAILURES:
                        print("❌ Mất tín hiệu quá lâu, dừng.")
                        break
                    time.sleep(CAM_RETRY_SLEEP)
                    continue
                retries = 0
                frame_count += 1

                # Cache GPS một lần/frame
                gps_data = gps.get()

                # Cảnh báo GPS stale
                if gps_data.get("stale"):
                    print(f"⚠ GPS stale — không cập nhật quá {GPS_STALE_SEC:.0f} giây!")

                # --- AI Inference ---
                results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

                pothole_detected = False
                max_conf         = 0.0
                annotated_frame  = frame

                if results:
                    r = results[0]
                    annotated_frame = r.plot()
                    if r.boxes is not None and len(r.boxes) > 0:
                        pothole_detected = True
                        max_conf = float(r.boxes.conf.max())

                # --- Logging ---
                if pothole_detected:
                    if not gps.ready:
                        print("⚠ Phát hiện ổ gà nhưng GPS chưa sẵn sàng — bỏ qua log.")
                    elif gps_data.get("stale"):
                        print("⚠ Phát hiện ổ gà nhưng GPS stale — bỏ qua log.")
                    else:
                        now         = time.monotonic()
                        time_ok     = (now - last_log_time) > MIN_LOG_INTERVAL_SEC

                        if last_logged_pos is None:
                            distance_ok = True
                            dist        = 0.0
                        else:
                            dist        = calculate_distance(
                                gps_data["lat"], gps_data["lon"],
                                *last_logged_pos,
                            )
                            distance_ok = dist >= MIN_LOG_DISTANCE_M

                        if time_ok and distance_ok:
                            last_log_time   = now
                            last_logged_pos = (gps_data["lat"], gps_data["lon"])

                            writer.writerow([
                                gps_data["timestamp_local"],
                                gps_data["lat"],
                                gps_data["lon"],
                                gps_data["speed"],
                                round(max_conf, 3),
                                frame_count,
                            ])
                            csv_f.flush()

                            emit_json_event(json_file, gps_data, max_conf, frame_count)

                            dist_str = f"{dist:.1f}m" if dist > 0 else "điểm đầu"
                            print(
                                f"📍 Ổ gà tại: {gps_data['lat']}, {gps_data['lon']} "
                                f"| Conf: {max_conf:.2f} | Cách điểm cũ: {dist_str}"
                            )

                # --- UI Overlay ---
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (400, 130), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

                cam_mode_label = "CAM:SNAP" if cap._mode == "snapshot" else "CAM:MJPG"
                gps_status     = "GPS:LOCK" if gps.ready else "GPS:WAIT"
                if gps_data.get("stale"):
                    gps_status = "GPS:STALE"

                cv2.putText(annotated_frame,
                            f"FPS:{fps:.1f}  {gps_status}  {cam_mode_label}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(annotated_frame,
                            f"LAT: {gps_data['lat']}  LON: {gps_data['lon']}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.putText(annotated_frame,
                            f"Speed: {gps_data['speed']} km/h",
                            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                if pothole_detected:
                    cv2.putText(annotated_frame,
                                f"!! POTHOLE  conf:{max_conf:.2f}",
                                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # Wrap imshow để tránh crash trên môi trường headless
                try:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("🛑 Đã nhận lệnh dừng từ người dùng.")
                        break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("🛑 Cửa sổ đã bị đóng.")
                        break
                except cv2.error:
                    print("⚠ Không thể hiển thị cửa sổ (headless?). Tiếp tục chạy không GUI.")
                    # Vẫn chạy inference + log, chỉ bỏ phần hiển thị
                    time.sleep(0.01)

        finally:
            cap.stop()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            print(f"\n👋 Hệ thống camera đã đóng an toàn.")
            print(f"📊 Tổng frames xử lý: {frame_count}")
            print(f"📁 CSV   : {log_file}")
            print(f"📁 JSONL : {json_file}")
            generate_map(log_file, log_dir, session_id)


if __name__ == "__main__":
    main()