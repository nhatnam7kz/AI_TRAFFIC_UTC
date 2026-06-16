"""
Pothole Detection — Real-time YOLO + IP Webcam + GPS
=====================================================
- Đã loại bỏ hoàn toàn Map Generator và CSV Logging.
- Đầu ra chuẩn hóa theo định dạng JSON cho Blockchain Dashboard.
- Tự động lấy đường dẫn tuyệt đối (Absolute Pathing).
"""

import cv2
import time
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
# TỰ ĐỘNG LẤY ĐƯỜNG DẪN GỐC CỦA FILE ĐỂ TRÁNH LỖI PATH
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# CONFIG
# ================================================================
PHONE_IP        = "192.168.1.78"
GPS_PORT        = 8081
CAM_PORT        = 8080

# Cập nhật đường dẫn linh hoạt
YOLO_WEIGHTS    = os.path.join(BASE_DIR, "weights/best.pt")
YOLO_IMGSZ      = 320
YOLO_CONF       = 0.25
LOG_DIR_BASE    = os.path.join(BASE_DIR, "logs")

# Chống log trùng lặp
MIN_LOG_INTERVAL_SEC  = 1.0
MIN_LOG_DISTANCE_M    = 3.0

# Retry camera
MAX_CONSECUTIVE_FAILURES = 30
CAM_RETRY_SLEEP          = 0.03

# ── FIX DELAY CAMERA ───────────────────────────────────────────
CAM_SNAPSHOT_MODE   = True
CAM_SNAPSHOT_FPS    = 30                           
CAM_THREAD_SLEEP    = 1.0 / CAM_SNAPSHOT_FPS   
CAM_SNAPSHOT_TIMEOUT = 2.0                      

# GPS stale detection
GPS_STALE_SEC = 5.0   


# ================================================================
# HAVERSINE ALGORITHM
# ================================================================
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ================================================================
# GPS — HTTP polling + stale detection
# ================================================================
class HTTPLocationGPS:
    def __init__(self, host: str, port: int = 8081) -> None:
        self.url          = f"http://{host}:{port}/gps"
        self._lat         = 21.027763
        self._lon         = 105.834160
        self._spd         = 0.0
        self.ready        = False
        self._lock        = Lock()
        self._last_update = 0.0

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
                pass # Bỏ qua in lỗi liên tục để Terminal sạch sẽ
            time.sleep(1.0)

    def get(self) -> dict:
        with self._lock:
            stale = (self.ready and (time.monotonic() - self._last_update) > GPS_STALE_SEC)
            return {
                "lat":             round(self._lat, 6),
                "lon":             round(self._lon, 6),
                "speed":           round(self._spd, 1),
                "stale":           stale,
            }


# ================================================================
# JSON EMITTER — Phù hợp với định dạng Dashboard Blockchain
# ================================================================
def emit_json_event(json_file: str, gps_data: dict, confidence: float) -> dict:
    event_data = {
        "lat": f"{gps_data['lat']:.4f}",
        "lng": f"{gps_data['lon']:.4f}",
        "type": "O_GA_NGUY_HIEM",
        "confidence": f"{confidence * 100:.1f}%",
        "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    }
    try:
        # Ghi append từng dòng (JSONL)
        with open(json_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠ Không ghi được JSON: {e}")
    
    return event_data


# ================================================================
# IP WEBCAM — snapshot mode
# ================================================================
class IPWebcamCapture:
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
            try:
                with urllib.request.urlopen(self._snapshot_url, timeout=CAM_SNAPSHOT_TIMEOUT) as resp:
                    if resp.status != 200: raise RuntimeError("Snapshot != 200")
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

    def _update_snapshot(self) -> None:
        while self._running:
            try:
                with urllib.request.urlopen(self._snapshot_url, timeout=CAM_SNAPSHOT_TIMEOUT) as resp:
                    raw = resp.read()
                arr = np.frombuffer(raw, dtype=np.uint8)
                frm = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                with self._lock:
                    self._success = frm is not None
                    self._frame   = frm
            except Exception as e:
                with self._lock: self._success = False
            time.sleep(CAM_THREAD_SLEEP)

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
            if self._frame is None: return False, None
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
    print("🔄 [IP WEBCAM + GPS MODE] Đang nạp mô hình...")
    model = YOLO(YOLO_WEIGHTS)
    print("✅ Mô hình đã sẵn sàng!")

    gps = HTTPLocationGPS(PHONE_IP, port=GPS_PORT).start()
    print("✅ GPS đã kết nối (đang chờ tín hiệu đầu tiên)...")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = os.path.join(LOG_DIR_BASE, f"live_{session_id}")
    os.makedirs(log_dir, exist_ok=True)
    json_file  = os.path.join(log_dir, "pothole_events.jsonl") 
    print(f"📁 Dữ liệu sự kiện sẽ lưu tại: {json_file}")

    print(f"🔗 Đang kết nối camera tới {PHONE_IP}:{CAM_PORT} ...")
    try:
        cap = IPWebcamCapture(PHONE_IP, CAM_PORT).start()
    except RuntimeError as e:
        print(e)
        return
    print("\n🚀 ĐANG CHẠY REAL-TIME + GPS THẬT... Ấn 'q' để THOÁT.\n")

    frame_count      = 0
    retries          = 0
    last_log_time    = 0.0
    last_logged_pos  = None
    window_name      = "YOLOv8 Pothole + GPS"
    t_last           = time.perf_counter()

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

            gps_data = gps.get()

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
                    pass # Chờ GPS sẵn sàng
                elif gps_data.get("stale"):
                    pass # Không log khi GPS bị stale
                else:
                    now         = time.monotonic()
                    time_ok     = (now - last_log_time) > MIN_LOG_INTERVAL_SEC

                    if last_logged_pos is None:
                        distance_ok = True
                    else:
                        dist = calculate_distance(gps_data["lat"], gps_data["lon"], *last_logged_pos)
                        distance_ok = dist >= MIN_LOG_DISTANCE_M

                    if time_ok and distance_ok:
                        last_log_time   = now
                        last_logged_pos = (gps_data["lat"], gps_data["lon"])

                        # Ghi log và in trực tiếp ra Terminal
                        event_data = emit_json_event(json_file, gps_data, max_conf)
                        print(f"🕳 CẢNH BÁO TẠI FRAME {frame_count}:")
                        print(json.dumps(event_data, indent=2, ensure_ascii=False))

            # --- UI Overlay ---
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

            cam_mode_label = "CAM:SNAP" if cap._mode == "snapshot" else "CAM:MJPG"
            gps_status     = "GPS:LOCK" if gps.ready else "GPS:WAIT"
            if gps_data.get("stale"): gps_status = "GPS:STALE"

            cv2.putText(annotated_frame, f"FPS:{fps:.1f}  {gps_status}  {cam_mode_label}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"LAT: {gps_data['lat']}  LON: {gps_data['lon']}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(annotated_frame, f"Speed: {gps_data['speed']} km/h",
                        (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            if pothole_detected:
                cv2.putText(annotated_frame, f"!! POTHOLE  conf:{max_conf:.2f}",
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            try:
                cv2.imshow(window_name, annotated_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("🛑 Đã nhận lệnh dừng từ người dùng.")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                time.sleep(0.01)

    finally:
        cap.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print(f"\n👋 Hệ thống camera đã đóng an toàn.")
        print(f"📊 Tổng frames xử lý: {frame_count}")
        print(f"📁 JSONL : {json_file}")

if __name__ == "__main__":
    main()