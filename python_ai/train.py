from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Tải mô hình YOLOv8n gốc làm nền tảng
    model = YOLO("weights/yolov8n.pt")

    # 2. Định vị file data.yaml trong thư mục dữ liệu đã mang sang
    data_yaml_path = os.path.abspath("python_ai/data_set/data.yaml")

    print(f"🔄 Đang kích hoạt GPU để huấn luyện...")

    # 3. Tiến hành huấn luyện bằng GPU xịn
    model.train(
        data=data_yaml_path,
        epochs=50,        # <--- TĂNG LÊN 50 hoặc 100 lượt vì GPU chạy siêu nhanh, giúp AI thông minh hơn nhiều!
        imgsz=640,        # <--- TRẢ VỀ CỬA SỔ 640 gốc (Ảnh nét nhất, nhận diện ổ gà chuẩn nhất từ xa)
        batch=16,         # <--- TĂNG BATCH lên 16 (GPU xử lý 16 ảnh cùng lúc thay vì 4 ảnh như CPU)
        device=0          # <--- BẮT BUỘC ĐỂ LÀ 0 để ép AI chạy bằng Card đồ họa NVIDIA rời
    )

    print("🎉 Quá trình huấn luyện bằng GPU trên máy mới kết thúc hoàn tất!")
    print("💡 Bộ não xịn nhất đã nằm tại: runs/detect/train/weights/best.pt")