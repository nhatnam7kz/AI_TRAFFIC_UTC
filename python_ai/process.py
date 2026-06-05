import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
import cv2

# Import tính năng từ các file module
from modules.video_extractor import VideoFrameExtractor
from modules.gps_processor import load_gps_from_gpx, extract_gps_from_video, interpolate_gps, haversine
from modules.detector import load_yolo_model, run_batch_inference
from modules.map_generator import generate_map

# ================================================================
# CONFIG — CHỈNH TÊN VIDEO TẠI ĐÂY
# ================================================================
VIDEO_NAME = "video2"       # <-- Điền tên video của bạn vào đây (không cần gõ đuôi .mp4)

# ================================================================
# CẤU HÌNH AI & HỆ THỐNG (Có thể giữ nguyên)
# ================================================================
YOLO_WEIGHTS       = "weights/best.pt"
YOLO_CONF          = 0.25
YOLO_BATCH_SIZE    = 4
PROCESS_FPS        = 2.0
DEVICE_ID          = "cam_001"

# Tự động map đường dẫn (File phải nằm trong thư mục 'sources/')
SOURCES_DIR = "sources"
VIDEO_PATH  = os.path.join(SOURCES_DIR, f"{VIDEO_NAME}.mp4")
GPX_PATH    = os.path.join(SOURCES_DIR, f"{VIDEO_NAME}.gpx")

# Cấu hình lưu log
YOLO_IMGSZ          = 320
MIN_LOG_DISTANCE_M  = 3.0
MIN_LOG_INTERVAL_S  = 1.0
SAVE_POTHOLE_FRAMES = True      
SAVE_SAMPLE_EVERY_N = 30        
FRAME_SAVE_QUALITY  = 85        
LOG_DIR_BASE        = "logs"

def main():
    print("====================================================")
    print(f"🎬 Đang chuẩn bị xử lý: '{VIDEO_NAME}'")
    print(f"📁 Video gốc : {VIDEO_PATH}")
    print(f"📍 Định vị   : {GPX_PATH}")
    print("====================================================")

    # Kiểm tra xem file video có tồn tại không
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ LỖI: Không tìm thấy file video tại: {VIDEO_PATH}")
        print("Vui lòng vứt file video vào thư mục 'sources/' và kiểm tra lại tên.")
        sys.exit(1)

    # 1. Nạp Model YOLO
    model = load_yolo_model(YOLO_WEIGHTS)

    # 2. Xử lý dữ liệu định vị (Ưu tiên file gpx cùng tên)
    gps_track = None
    if os.path.exists(GPX_PATH):
        gps_track = load_gps_from_gpx(GPX_PATH)
    else:
        print(f"⚠ Không thấy file GPX tại {GPX_PATH}. Thử trích xuất từ metadata video...")
        gps_track = extract_gps_from_video(VIDEO_PATH)

    if not gps_track:
        print("⚠ Không tìm thấy bất kỳ nguồn GPS nào — dùng toạ độ mặc định (0, 0).")
        gps_track = [{"time_s": 0, "lat": 0.0, "lon": 0.0, "speed": 0.0}]

    # 3. Khởi tạo cây thư mục Logs
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = os.path.join(LOG_DIR_BASE, f"{VIDEO_NAME}_{session_id}")
    frames_dir = os.path.join(log_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    log_file  = os.path.join(log_dir, "pothole_log.csv")
    json_file = os.path.join(log_dir, "pothole_events.jsonl")

    # 4. Đọc luồng Video
    try:
        extractor = VideoFrameExtractor(VIDEO_PATH, process_fps=PROCESS_FPS)
    except RuntimeError as e:
        print(e)
        return

    # 5. Core xử lý chính theo Batch
    t_start = time.perf_counter()
    processed_count = 0
    state = {"last_log_time_s": -999.0, "last_logged_pos": None, "pothole_count": 0}

    batch_frames, batch_meta = [], []
    csv_rows, json_events = [], []

    def flush_batch():
        if not batch_frames: return
        results = run_batch_inference(model, batch_frames, YOLO_IMGSZ, YOLO_CONF)

        for (f_idx, t_s, raw_frame), result in zip(batch_meta, results):
            gps = interpolate_gps(gps_track, t_s)
            ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            pothole_here = False
            max_conf     = 0.0
            annotated    = raw_frame

            if result.boxes is not None and len(result.boxes) > 0:
                pothole_here = True
                max_conf     = float(result.boxes.conf.max())
                annotated    = result.plot()

            time_ok = (t_s - state["last_log_time_s"]) >= MIN_LOG_INTERVAL_S
            dist_ok = True
            dist = 0.0
            if state["last_logged_pos"]:
                dist = haversine(gps["lat"], gps["lon"], *state["last_logged_pos"])
                dist_ok = dist >= MIN_LOG_DISTANCE_M

            if pothole_here and time_ok and dist_ok:
                state["pothole_count"] += 1
                state["last_log_time_s"] = t_s
                state["last_logged_pos"] = (gps["lat"], gps["lon"])

                frame_filename = f"pothole_{f_idx:06d}_conf{max_conf:.2f}.jpg" if SAVE_POTHOLE_FRAMES else ""
                if frame_filename:
                    cv2.imwrite(os.path.join(frames_dir, frame_filename), annotated, [cv2.IMWRITE_JPEG_QUALITY, FRAME_SAVE_QUALITY])

                csv_rows.append([ts_local, round(gps["lat"], 6), round(gps["lon"], 6), round(gps["speed"], 1), round(max_conf, 3), f_idx, frame_filename])
                json_events.append({
                    "lat": round(gps["lat"], 6), "lng": round(gps["lon"], 6), "type": "pothole", "confidence": round(max_conf, 3),
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "device_id": DEVICE_ID,
                    "speed_kmh": round(gps["speed"], 1), "frame": f_idx, "frame_file": frame_filename,
                })
                print(f"  🕳  Ổ gà mới phát hiện tại t={t_s:6.1f}s | độ tin cậy conf={max_conf:.2f}")

            elif SAVE_SAMPLE_EVERY_N > 0 and (processed_count % SAVE_SAMPLE_EVERY_N == 0):
                cv2.imwrite(os.path.join(frames_dir, f"sample_{f_idx:06d}.jpg"), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_SAVE_QUALITY - 10])

        batch_frames.clear()
        batch_meta.clear()

    print("\n🚀 Bắt đầu xử lý luồng hệ thống...\n")
    for frame_idx, time_s, frame in extractor:
        processed_count += 1
        batch_frames.append(frame)
        batch_meta.append((frame_idx, time_s, frame))
        if len(batch_frames) >= YOLO_BATCH_SIZE: flush_batch()

    flush_batch()
    extractor.release()

    # 6. Ghi dữ liệu báo cáo đầu ra
    if state['pothole_count'] > 0:
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "latitude", "longitude", "speed_kmh", "confidence", "frame", "frame_file"])
            writer.writerows(csv_rows)

        with open(json_file, "w", encoding="utf-8") as f:
            for event in json_events: f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # 7. Render giao diện Web Map
        generate_map(log_file, log_dir, session_id)
    else:
        print("✨ Tuyệt vời! Không phát hiện thấy ổ gà nào trên đoạn video này.")

    elapsed = time.perf_counter() - t_start
    print(f"\n╔══════════════════════════════════════════════╗\n  ✅ HOÀN THÀNH DỰ ÁN\n  ⏱  Xử lý: {elapsed:.1f}s | Tốc độ: {processed_count/elapsed:.1f} fps\n  🕳  Tổng ổ gà phát hiện: {state['pothole_count']}\n  📁 Folder kết quả: {log_dir}\n╚══════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()