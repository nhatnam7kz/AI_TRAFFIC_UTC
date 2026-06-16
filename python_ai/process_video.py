"""
Pothole Detection — Offline Video Mode
=====================================================
- Đã loại bỏ hoàn toàn Map Generator và CSV Logging.
- Đầu ra chuẩn hóa theo định dạng JSON cho Blockchain Dashboard.
- Tự động lấy đường dẫn tuyệt đối (Absolute Pathing).
- Ghi log dạng JSONL (append-only) giống IP Webcam.
"""

import json
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# ================================================================
# TỰ ĐỘNG LẤY ĐƯỜNG DẪN GỐC CỦA FILE ĐỂ TRÁNH LỖI PATH
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import module xử lý dữ liệu (Đã loại bỏ map_generator)
from modules.gps_processor import load_gps_from_gpx, extract_gps_from_video, interpolate_gps, haversine

# ================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ================================================================
YOLO_WEIGHTS       = os.path.join(BASE_DIR, "weights/best.pt")
YOLO_CONF          = 0.25
DEVICE_ID          = "cam_001"
SOURCES_DIR        = os.path.join(BASE_DIR, "sources")

YOLO_IMGSZ         = 320
MIN_LOG_DISTANCE_M = 3.0
MIN_LOG_INTERVAL_S = 1.0
SAVE_POTHOLE_FRAMES= True
FRAME_SAVE_QUALITY = 85
LOG_DIR_BASE       = os.path.join(BASE_DIR, "logs")

# Thêm tuỳ chọn bỏ qua khung hình (Frame Skip) giúp Edge-AI chạy mượt hơn
PROCESS_EVERY_N_FRAMES = 2 

# ================================================================
# HÀM GHI JSON SỰ KIỆN
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
        # Ghi append từng dòng (JSONL) để đồng bộ với bản Webcam
        with open(json_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠ Không ghi được JSON: {e}")
    
    return event_data

# ================================================================
# 2. HÀM XỬ LÝ CHÍNH
# ================================================================
def process_video_smooth(model, video_path: str, gpx_path: str, video_name: str):
    print(f"\n{'='*52}")
    print(f"🎬 ĐANG XỬ LÝ : {video_name}")
    print(f"📁 Đường dẫn : {video_path}")
    print(f"📍 GPX       : {gpx_path}")
    print(f"{'='*52}")

    # --- NẠP DỮ LIỆU GPS CÓ BẢO VỆ LỖI ---
    try:
        if os.path.exists(gpx_path):
            gps_track = load_gps_from_gpx(gpx_path)
        else:
            print(f"⚠ Không thấy GPX → thử trích từ metadata video...")
            gps_track = extract_gps_from_video(video_path)

        if not gps_track:
            print("⚠ Không có GPS → dùng toạ độ mặc định (0, 0).")
            gps_track = [{"time_s": 0, "lat": 0.0, "lon": 0.0, "speed": 0.0}]
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu GPS: {e}")
        return

    # --- TẠO THƯ MỤC LOG ---
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir    = os.path.join(LOG_DIR_BASE, f"{video_name}_{session_id}")
    frames_dir = os.path.join(log_dir, "frames")
    json_file  = os.path.join(log_dir, "pothole_events.jsonl") # Sử dụng JSONL

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

    frame_idx = 0
    t_start = time.perf_counter()

    window_name = f"Edge-AI Live View - {video_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Đảm bảo tạo thư mục log trước khi chạy
    os.makedirs(log_dir, exist_ok=True)

    # ================================================================
    # VÒNG LẶP XỬ LÝ TỪNG KHUNG HÌNH
    # ================================================================
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Bỏ qua frame theo tham số tối ưu (Frame Skip)
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue
        
        t0_infer = time.perf_counter()
        
        # 1. Chạy AI trên khung hình hiện tại
        results = model(frame, stream=True, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        annotated_frame = frame 
        
        pothole_here = False
        max_conf = 0.0

        for r in results:
            annotated_frame = r.plot()
            if r.boxes is not None and len(r.boxes) > 0:
                pothole_here = True
                max_conf = float(r.boxes.conf.max())

        # 2. Xử lý logic định vị và lưu JSON nếu có ổ gà
        t_s = frame_idx / video_fps

        if pothole_here:
            try:
                # Chỉ tính toán GPS khi phát hiện vật thể
                gps = interpolate_gps(gps_track, t_s)
                
                # Gom gọn logic tính khoảng cách
                time_ok = (t_s - state["last_log_time_s"]) >= MIN_LOG_INTERVAL_S
                dist_ok = (not state["last_logged_pos"]) or \
                          (haversine(gps["lat"], gps["lon"], *state["last_logged_pos"]) >= MIN_LOG_DISTANCE_M)

                if time_ok and dist_ok:
                    state["pothole_count"] += 1
                    state["last_log_time_s"] = t_s
                    state["last_logged_pos"] = (gps["lat"], gps["lon"])

                    # Khởi tạo thư mục frames chỉ khi có ổ gà
                    if SAVE_POTHOLE_FRAMES:
                        os.makedirs(frames_dir, exist_ok=True)
                        frame_filename = f"pothole_{frame_idx:06d}_conf{max_conf:.2f}.jpg"
                        cv2.imwrite(os.path.join(frames_dir, frame_filename), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_SAVE_QUALITY])

                    # Ghi JSONL và in ra màn hình
                    event_data = emit_json_event(json_file, gps, max_conf)
                    print(f"🕳 CẢNH BÁO TẠI FRAME {frame_idx}:")
                    print(json.dumps(event_data, indent=2, ensure_ascii=False))

            except Exception as e:
                print(f"❌ Lỗi khi xử lý GPS hoặc log ở frame {frame_idx}: {e}")

        # 3. Hiển thị thông số 
        elapsed = time.perf_counter() - t0_infer
        fps_display = int(1 / elapsed) if elapsed > 0 else 999 
        
        color = (0, 0, 255) if pothole_here else (0, 255, 0)
        
        cv2.putText(annotated_frame, f"File: {video_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps_display}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Potholes: {state['pothole_count']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        try:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\n⏭️ Bỏ qua video: {video_name}")
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            time.sleep(0.01)

        frame_idx += 1

    # Dọn dẹp
    cap.release()
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass

    # --- 4. TỔNG KẾT ---
    elapsed_total = time.perf_counter() - t_start
    print(f"\n✅ Xong [{video_name}] — {elapsed_total:.1f}s | 🕳 {state['pothole_count']} ổ gà ghi nhận.")
    if state["pothole_count"] > 0:
        print(f"📁 JSONL : {json_file}\n")


# ================================================================
# HÀM MAIN
# ================================================================
def main():
    print("====================================================")
    print("🚗 Decentralized Edge-AI — Core Detection Mode")
    print(f"📂 Quét thư mục: {SOURCES_DIR}")
    print("====================================================\n")

    if not os.path.isdir(SOURCES_DIR):
        print(f"❌ Không tìm thấy thư mục '{SOURCES_DIR}'.")
        sys.exit(1)

    video_files = sorted([f for f in os.listdir(SOURCES_DIR) if f.lower().endswith(".mp4")])

    if not video_files:
        print(f"❌ Không tìm thấy file .mp4 nào trong '{SOURCES_DIR}'.")
        sys.exit(1)

    print("🔄 Đang nạp mô hình YOLO...")
    try:
        model = YOLO(YOLO_WEIGHTS)
        # Warm-up (Chạy mồi một ma trận rỗng) để ngăn lag JIT Compile
        _ = model(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8))
        print("✅ Nạp & Warm-up xong!")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình AI: {e}")
        sys.exit(1)

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