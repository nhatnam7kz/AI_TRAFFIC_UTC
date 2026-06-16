import sys

def load_yolo_model(weights_path: str):
    print("🔄 Đang nạp mô hình YOLO...")
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"✅ Model nạp thành công: {weights_path}")
        return model
    except Exception as e:
        print(f"❌ Không nạp được model: {e}")
        sys.exit(1)

def run_batch_inference(model, frames: list, imgsz: int, conf: float, half: bool = False) -> list:
    if not frames:
        return []
    results = model(frames, imgsz=imgsz, conf=conf, half=half, verbose=False, stream=False)
    return list(results)