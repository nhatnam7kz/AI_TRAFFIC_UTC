"""
Fake GPS Track Generator for UTC (Đại học Giao thông Vận tải)
=============================================================
Nhiệm vụ: Tạo ra file dữ liệu định vị .gpx giả lập hành trình di chuyển
để khớp (nội suy) mốc thời gian thực với các khung hình trong video hành trình.
"""

import datetime

# --- Cấu hình dữ liệu Fake tại UTC Giao Thông Vận Tải ---
TOTAL_SECONDS = 60      # Thời lượng bằng với video của bạn (giây)
START_LAT = 21.025686   # Tọa độ ĐH Giao thông Vận tải (Cổng chính/Cầu vượt đi bộ)
START_LON = 105.802165
SPEED_KMH = 35.0        # Tốc độ di chuyển giả lập (km/h)
# --------------------------------------------------------

# Chuyển đổi tốc độ từ km/h sang độ (vĩ độ/kinh độ) di chuyển mỗi giây
speed_mps = SPEED_KMH / 3.6
delta_deg_per_sec = (speed_mps * 0.000009)

gpx_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="FakeGPXGenerator" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Fake Route for Dashcam</name>
    <trkseg>"""

now = datetime.datetime.now(datetime.timezone.utc)

for i in range(TOTAL_SECONDS + 1):
    # Mỗi giây xe sẽ tịnh tiến sang phải/lên trên một chút tạo thành một đường thẳng
    current_lat = START_LAT + (i * delta_deg_per_sec * 0.7)
    current_lon = START_LON + (i * delta_deg_per_sec * 0.7)
    current_time = (now + datetime.timedelta(seconds=i)).isoformat()

    gpx_content += f"""
      <trkpt lat="{current_lat:.6f}" lon="{current_lon:.6f}">
        <time>{current_time}</time>
        <speed>{speed_mps:.2f}</speed>
      </trkpt>"""

gpx_content += """
    </trkseg>
  </trk>
</gpx>"""

# ĐÃ SỬA: Ghi trực tiếp vào thư mục hiện tại
with open("fake_track.gpx", "w", encoding="utf-8") as f:
    f.write(gpx_content)

print("✅ Đã tạo xong file GPS giả lập tại: fake_track.gpx")