import os
import csv

def generate_map(log_file: str, out_dir: str, session_id: str) -> None:
    points = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"timestamp", "latitude", "longitude", "speed_kmh", "confidence"}
            for row in reader:
                if not required.issubset(row.keys()) or not row["latitude"] or not row["longitude"]:
                    continue
                try:
                    float(row["latitude"])
                    float(row["longitude"])
                    float(row["confidence"])
                except ValueError:
                    continue
                points.append(row)
    except Exception as e:
        print(f"⚠ Không đọc được log để vẽ map: {e}")
        return

    if not points:
        print("⚠ Không có ổ gà hợp lệ nào để hiển thị lên bản đồ.")
        return

    center_lat = sum(float(p["latitude"])  for p in points) / len(points)
    center_lon = sum(float(p["longitude"]) for p in points) / len(points)

    markers_js = ""
    for p in points:
        try:
            lat, lon, conf = float(p["latitude"]), float(p["longitude"]), float(p["confidence"])
        except (ValueError, TypeError): continue
            
        color = "green" if conf >= 0.6 else "orange" if conf >= 0.4 else "red"
        frame_img = p.get("frame_file", "")
        popup_img = f'<br><img src="frames/{frame_img}" width="200" style="border-radius:4px">' if frame_img else ""
                     
        markers_js += f"""
        L.circleMarker([{lat}, {lon}], {{
            color: '{color}', fillColor: '{color}', fillOpacity: 0.8, radius: 8
        }}).addTo(map).bindPopup(
            "<b>&#9888; &#7893; g&#224;</b><br>" +
            "&#128336; {p['timestamp']}<br>" +
            "&#128663; {p['speed_kmh']} km/h<br>" +
            "&#127919; Conf: {conf:.3f}" + '{popup_img}'
        );
        """

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <title>Bản đồ ổ gà — {session_id}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; }}
        #info {{
            position: absolute; top: 10px; left: 50px; z-index: 1000;
            background: white; padding: 10px 16px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-size: 14px;
        }}
    </style>
</head>
<body>
<div id="info">
    📍 Tổng số ổ gà: <b>{len(points)}</b> &nbsp;|&nbsp; Session: <b>{session_id}</b> &nbsp;|&nbsp;
    <span style="color:green">● conf&ge;0.6</span> &nbsp; <span style="color:orange">● 0.4-0.6</span> &nbsp; <span style="color:red">● &lt;0.4</span>
</div>
<div id="map"></div>
<script>
    var map = L.map('map').setView([{center_lat}, {center_lon}], 16);
    L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{ attribution: '&copy; Google Maps' }}).addTo(map);
    {markers_js}
</script>
</body>
</html>"""

    out = os.path.join(out_dir, "map.html")
    with open(out, "w", encoding="utf-8") as f: f.write(html)
    print(f"🗺  Map: {out}")