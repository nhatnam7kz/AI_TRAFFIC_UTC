import json
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Tự động tìm đường dẫn tuyệt đối của thư mục chứa file code này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import module xử lý dữ liệu
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
    # Thêm microsecond vào ID session để tránh trùng thư mục
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir    = os.path.join(LOG_DIR_BASE, f"{video_name}_{session_id}")
    frames_dir = os.path.join(log_dir, "frames")
    json_file  = os.path.join(log_dir, "pothole_events.json")

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

    json_events = []
    frame_idx = 0
    t_start = time.perf_counter()

    window_name = f"Edge-AI Live View - {video_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

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
        
        # 1. Chạy AI trên khung hình hiện tại (Bỏ frame.copy() thừa)
        results = model(frame, stream=True, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
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
                
                # Gom gọn logic tính khoảng cách (Short-circuit an toàn)
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

                    # Format theo yêu cầu Blockchain/Dashboard 
                    event_data = {
                        "lat": f"{gps['lat']:.4f}",
                        "lng": f"{gps['lon']:.4f}",
                        "type": "O_GA_NGUY_HIEM",
                        "confidence": f"{max_conf * 100:.1f}%",
                        "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                    }
                    
                    json_events.append(event_data)
                    
                    # Trả về chuỗi JSON trực tiếp
                    print(f"🕳 CẢNH BÁO TẠI FRAME {frame_idx}:")
                    print(json.dumps(event_data, indent=2, ensure_ascii=False))

            except Exception as e:
                print(f"❌ Lỗi khi xử lý GPS hoặc log ở frame {frame_idx}: {e}")

        # 3. Hiển thị thông số 
        elapsed = time.perf_counter() - t0_infer
        fps_display = int(1 / elapsed) if elapsed > 0 else 999  # Sửa ZeroDivisionError
        
        # Chỉ chuyển thành màu đỏ khi ổ gà nằm trên khung hình
        color = (0, 0, 255) if pothole_here else (0, 255, 0)
        
        cv2.putText(annotated_frame, f"File: {video_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {fps_display}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Potholes: {state['pothole_count']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n⏭️ Bỏ qua video: {video_name}")
            break

        frame_idx += 1

    # Dọn dẹp
    cap.release()
    cv2.destroyWindow(window_name)

    # --- 4. GHI LOG SỰ KIỆN KHI KẾT THÚC ---
    if state["pothole_count"] > 0:
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_events, f, ensure_ascii=False, indent=4)
            print(f"✅ Đã xuất {len(json_events)} sự kiện ra file: {json_file}")
        except Exception as e:
            print(f"❌ Lỗi ghi file JSON: {e}")
    else:
        print("✨ Không phát hiện ổ gà nào trong video này.")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n✅ Xong [{video_name}] — {elapsed_total:.1f}s | 🕳 {state['pothole_count']} ổ gà ghi nhận.\n")


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