import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
import cv2
from ultralytics import YOLO

# Import các module xử lý dữ liệu (Bỏ module batching và threading)
from modules.gps_processor import load_gps_from_gpx, extract_gps_from_video, interpolate_gps, haversine
from modules.map_generator import generate_map

# ================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ================================================================
YOLO_WEIGHTS       = "weights/best.pt"
YOLO_CONF          = 0.25
DEVICE_ID          = "cam_001"
SOURCES_DIR        = "sources"

# Cấu hình log dữ liệu (Giữ nguyên để không spam báo cáo trùng lặp)
YOLO_IMGSZ          = 320
MIN_LOG_DISTANCE_M  = 3.0   # Cách nhau 3 mét mới tính là 1 ổ gà mới
MIN_LOG_INTERVAL_S  = 1.0   # Cách nhau 1 giây mới log lại
SAVE_POTHOLE_FRAMES = True
FRAME_SAVE_QUALITY  = 85
LOG_DIR_BASE        = "logs"

# ================================================================
# 2. HÀM XỬ LÝ CHÍNH
# ================================================================
def process_video_smooth(model, video_path: str, gpx_path: str, video_name: str):
    print(f"\n{'='*52}")
    print(f"🎬 ĐANG XỬ LÝ : {video_name}")
    print(f"📁 Đường dẫn : {video_path}")
    print(f"📍 GPX       : {gpx_path}")
    print(f"{'='*52}")

    # --- NẠP DỮ LIỆU GPS ---
    gps_track = None
    if os.path.exists(gpx_path):
        gps_track = load_gps_from_gpx(gpx_path)
    else:
        print(f"⚠ Không thấy GPX → thử trích từ metadata video...")
        gps_track = extract_gps_from_video(video_path)

    if not gps_track:
        print("⚠ Không có GPS → dùng toạ độ mặc định (0, 0).")
        gps_track = [{"time_s": 0, "lat": 0.0, "lon": 0.0, "speed": 0.0}]

    # --- TẠO THƯ MỤC LOG ---
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = os.path.join(LOG_DIR_BASE, f"{video_name}_{session_id}")
    frames_dir = os.path.join(log_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    log_file  = os.path.join(log_dir, "pothole_log.csv")
    json_file = os.path.join(log_dir, "pothole_events.jsonl")

    # --- KHỞI TẠO VIDEO ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Định dạng: {video_fps:.1f} FPS | Tổng: {total_frames} frames\n")

    # --- BIẾN TRẠNG THÁI ---
    state = {
        "last_log_time_s":  -999.0,
        "last_logged_pos":  None,
        "pothole_count":    0,
    }

    csv_rows, json_events = [], []
    frame_idx = 0
    t_start = time.perf_counter()

    # Mở cửa sổ hiển thị
    window_name = f"Edge-AI Live View - {video_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # ================================================================
    # VÒNG LẶP XỬ LÝ TỪNG KHUNG HÌNH (MƯỢT NHƯ CODE 1)
    # ================================================================
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        t0_infer = time.perf_counter()
        
        # 1. Chạy AI trên khung hình hiện tại
        results = model(frame, stream=True, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
        annotated_frame = frame.copy()
        
        pothole_here = False
        max_conf = 0.0

        for r in results:
            annotated_frame = r.plot() # AI tự vẽ Box vào ảnh
            if r.boxes is not None and len(r.boxes) > 0:
                pothole_here = True
                max_conf = float(r.boxes.conf.max())

        # 2. Xử lý Logic định vị và lưu log nếu có ổ gà
        t_s = frame_idx / video_fps
        gps = interpolate_gps(gps_track, t_s)

        time_ok = (t_s - state["last_log_time_s"]) >= MIN_LOG_INTERVAL_S
        dist_ok = True
        if state["last_logged_pos"]:
            dist = haversine(gps["lat"], gps["lon"], *state["last_logged_pos"])
            dist_ok = dist >= MIN_LOG_DISTANCE_M

        if pothole_here and time_ok and dist_ok:
            state["pothole_count"] += 1
            state["last_log_time_s"] = t_s
            state["last_logged_pos"] = (gps["lat"], gps["lon"])
            ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Lưu ảnh
            frame_filename = f"pothole_{frame_idx:06d}_conf{max_conf:.2f}.jpg" if SAVE_POTHOLE_FRAMES else ""
            if frame_filename:
                cv2.imwrite(os.path.join(frames_dir, frame_filename), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_SAVE_QUALITY])

            # Lưu data vào List
            csv_rows.append([ts_local, round(gps["lat"], 6), round(gps["lon"], 6), round(gps["speed"], 1), round(max_conf, 3), frame_idx, frame_filename])
            json_events.append({
                "lat": round(gps["lat"], 6), "lng": round(gps["lon"], 6), "type": "pothole", "confidence": round(max_conf, 3),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "device_id": DEVICE_ID,
                "speed_kmh": round(gps["speed"], 1), "frame": frame_idx, "frame_file": frame_filename,
            })
            print(f"  🕳 Ổ gà #{state['pothole_count']} phát hiện tại t={t_s:6.1f}s | conf={max_conf:.2f}")

        # 3. Hiển thị thông số mượt mà lên màn hình
        fps_display = int(1 / (time.perf_counter() - t0_infer))
        color = (0, 0, 255) if pothole_here else (0, 255, 0)
        
        cv2.putText(annotated_frame, f"File: {video_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps_display}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Potholes: {state['pothole_count']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(window_name, annotated_frame)

        # Chờ 1ms, bấm 'q' để bỏ qua video này
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n⏭️ Bỏ qua video: {video_name}")
            break

        frame_idx += 1

    # Dọn dẹp video hiện tại
    cap.release()
    cv2.destroyWindow(window_name)

    # --- 4. GHI LOG VÀ RENDER MAP KHI KẾT THÚC VIDEO ---
    if state["pothole_count"] > 0:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "latitude", "longitude", "speed_kmh", "confidence", "frame", "frame_file"])
            writer.writerows(csv_rows)

        with open(json_file, "w", encoding="utf-8") as f:
            for event in json_events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        print("🗺️ Đang khởi tạo bản đồ kết quả...")
        generate_map(log_file, log_dir, session_id)
        print(f"✅ Đã xuất bản đồ tại thư mục: {log_dir}")
    else:
        print("✨ Không phát hiện ổ gà nào trong video này.")

    elapsed = time.perf_counter() - t_start
    print(f"\n✅ Xong [{video_name}] — {elapsed:.1f}s | 🕳 {state['pothole_count']} ổ gà ghi nhận.\n")


# ================================================================
# HÀM MAIN - QUÉT TOÀN BỘ VIDEO TRONG THƯ MỤC
# ================================================================
def main():
    print("====================================================")
    print("🚗 Decentralized Edge-AI — Smooth Detector Mode")
    print(f"📂 Quét thư mục: {os.path.abspath(SOURCES_DIR)}")
    print("====================================================\n")

    if not os.path.isdir(SOURCES_DIR):
        print(f"❌ Không tìm thấy thư mục '{SOURCES_DIR}'.")
        sys.exit(1)

    video_files = sorted([f for f in os.listdir(SOURCES_DIR) if f.lower().endswith(".mp4")])

    if not video_files:
        print(f"❌ Không tìm thấy file .mp4 nào trong '{SOURCES_DIR}'.")
        sys.exit(1)

    print("🔄 Đang nạp mô hình YOLO...")
    model = YOLO(YOLO_WEIGHTS)
    print("✅ Nạp xong!")

    # Duyệt qua từng video
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(SOURCES_DIR, video_file)
        gpx_path   = os.path.join(SOURCES_DIR, f"{video_name}.gpx")
        
        process_video_smooth(model, video_path, gpx_path, video_name)

    cv2.destroyAllWindows()
    print("\n╔══════════════════════════════════════════════╗")
    print("  🎉 Hoàn thành toàn bộ danh sách video!")
    print("╚══════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()