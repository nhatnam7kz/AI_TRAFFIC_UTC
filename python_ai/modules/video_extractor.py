import cv2

class VideoFrameExtractor:
    """Đọc video và yield (frame_idx, time_s, frame) tại đúng FPS mong muốn."""
    def __init__(self, video_path: str, process_fps: float = 2.0):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Không mở được video: {video_path}")

        self.video_fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s  = self.total_frames / self.video_fps
        self.step        = max(1, int(self.video_fps / process_fps))

        total_process = self.total_frames // self.step
        print(f"📹 Video: {self.total_frames} frames | {self.video_fps:.1f} fps | {self.duration_s:.1f}s")
        print(f"⚡ Xử lý: mỗi {self.step} frame (~{process_fps} fps) → ~{total_process} frames cần inference")

    def __iter__(self):
        frame_idx = 0
        while frame_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
            if not ok:
                break
            time_s = frame_idx / self.video_fps
            yield frame_idx, time_s, frame
            frame_idx += self.step

    def release(self):
        self.cap.release()