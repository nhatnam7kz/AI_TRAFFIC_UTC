#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>  // FIX 1: dùng atomic thay bool thường

// ================================================================
// 1. CẤU TRÚC DỮ LIỆU FRAME
// ================================================================
struct FrameBuffer {
    int width = 640;
    int height = 640;
    int channels = 3;
    std::vector<uint8_t> data;

    FrameBuffer() {
        data.resize(width * height * channels, 0);
    }

    // FIX 2: Move constructor — cho phép std::move() không copy data
    FrameBuffer(FrameBuffer&&) = default;
    FrameBuffer& operator=(FrameBuffer&&) = default;

    // Xóa copy constructor — ngăn vô tình copy 1.2MB
    FrameBuffer(const FrameBuffer&) = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;
};

// ================================================================
// 2. BIẾN DÙNG CHUNG GIỮA 2 LUỒNG
// ================================================================
std::queue<FrameBuffer> frameQueue;
std::mutex mtx;
std::condition_variable cvNotEmpty;   // Báo AIThread: có frame mới
std::condition_variable cvNotFull;    // Báo CameraThread: queue còn chỗ

// FIX 1: atomic<bool> — an toàn khi đọc/ghi từ 2 thread khác nhau
std::atomic<bool> isRunning{true};

constexpr int MAX_QUEUE_SIZE = 5;  // FIX 3: giới hạn queue tránh tràn RAM

// ================================================================
// 3. LUỒNG ĐỌC CAMERA
// ================================================================
void CameraCaptureThread() {
    int frameCount = 0;

    while (frameCount < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // Giả lập 30 FPS

        FrameBuffer newFrame;

        {
            std::unique_lock<std::mutex> lock(mtx);

            // FIX 3: Nếu queue đầy thì chờ AI xử lý bớt — không push thêm
            cvNotFull.wait(lock, [] {
                return frameQueue.size() < MAX_QUEUE_SIZE || !isRunning;
            });

            if (!isRunning) break;

            // FIX 2: std::move — chuyển ownership, không copy 1.2MB
            frameQueue.push(std::move(newFrame));
        }

        cvNotEmpty.notify_one();
        frameCount++;
    }

    // FIX 1: atomic — không cần lock khi set isRunning
    isRunning = false;
    cvNotEmpty.notify_all();  // Đánh thức AIThread để thoát
    cvNotFull.notify_all();
}

// ================================================================
// 4. LUỒNG AI INFERENCE
// ================================================================
void AIInferenceThread() {
    auto start_time = std::chrono::high_resolution_clock::now();
    int processedFrames = 0;

    while (true) {
        FrameBuffer currentFrame;

        {
            std::unique_lock<std::mutex> lock(mtx);

            cvNotEmpty.wait(lock, [] {
                return !frameQueue.empty() || !isRunning;
            });

            if (frameQueue.empty() && !isRunning) break;

            // FIX 2: std::move — lấy frame không copy
            currentFrame = std::move(frameQueue.front());
            frameQueue.pop();
        }

        // Báo CameraThread: queue đã có chỗ trống
        cvNotFull.notify_one();

        // Giả lập AI inference 10ms (INT8 quantized)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        processedFrames++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    double seconds = duration.count() / 1000.0;

    std::cout << "\n================ KẾT QUẢ TỐI ƯU C++ ================" << std::endl;
    std::cout << "Tổng frames xử lý : " << processedFrames << std::endl;
    std::cout << "Tổng thời gian     : " << seconds << " giây" << std::endl;
    std::cout << "Hiệu năng          : " << (processedFrames / seconds) << " FPS" << std::endl;
    std::cout << "Queue size tối đa  : " << MAX_QUEUE_SIZE << " frames" << std::endl;
}

// ================================================================
// 5. MAIN
// ================================================================
int main() {
    std::cout << "🔄 Khởi chạy pipeline Camera → AI (C++ optimized)..." << std::endl;

    std::thread t1(CameraCaptureThread);
    std::thread t2(AIInferenceThread);

    t1.join();
    t2.join();

    std::cout << "✅ Pipeline kết thúc an toàn." << std::endl;
    return 0;
}