import pandas as pd
import openrouteservice
import folium

# ===== 1. Nhập API Key từ OpenRouteService =====
API_KEY = '5b3ce3597851110001cf6248b00ac6acf0fc43ec93ea37239079ce59'  # ← Thay bằng API thật từ https://openrouteservice.org/dev/#/signup

# ===== 2. Đọc dữ liệu Excel =====
df = pd.read_excel("D:\\BachHoaXanh_Data.xlsx")

# Gán lại tên cột để dễ truy cập
df.columns = ['STT', 'Dia_diem', 'Kinh_do', 'Vi_do', 'Thoi_gian', 'Mat_hang']

# ===== 3. Chọn địa điểm cần tìm đường đi =====
dia_diem_1 = "Trương Vĩnh Ký"
dia_diem_2 = "An Dương Vương"

# ===== 4. Tìm dòng khớp địa điểm =====
try:
    diem_1 = df[df['Dia_diem'].str.contains(dia_diem_1, case=False, na=False)].iloc[0]
    diem_2 = df[df['Dia_diem'].str.contains(dia_diem_2, case=False, na=False)].iloc[0]
except IndexError:
    print("❌ Không tìm thấy địa điểm khớp! Vui lòng kiểm tra lại tên.")
    exit()

# Tọa độ theo định dạng [Kinh độ, Vĩ độ]
start_coords = [diem_1['Kinh_do'], diem_1['Vi_do']]
end_coords = [diem_2['Kinh_do'], diem_2['Vi_do']]

# ===== 5. Gọi API OpenRouteService để tính đường đi =====
client = openrouteservice.Client(key=API_KEY)

route = client.directions(
    coordinates=[start_coords, end_coords],
    profile='driving-car',
    format='geojson'
)

# ===== 6. Tạo bản đồ với tuyến đường =====
center_lat = (start_coords[1] + end_coords[1]) / 2
center_lon = (start_coords[0] + end_coords[0]) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Marker điểm bắt đầu
folium.Marker(
    location=[start_coords[1], start_coords[0]],
    popup=folium.Popup(f"""
        <b>{diem_1['Dia_diem']}</b><br>
        🕒 {diem_1['Thoi_gian']}<br>
        📦 {diem_1['Mat_hang']}
    """, max_width=300),
    icon=folium.Icon(color='green')
).add_to(m)

# Marker điểm đến
folium.Marker(
    location=[end_coords[1], end_coords[0]],
    popup=folium.Popup(f"""
        <b>{diem_2['Dia_diem']}</b><br>
        🕒 {diem_2['Thoi_gian']}<br>
        📦 {diem_2['Mat_hang']}
    """, max_width=300),
    icon=folium.Icon(color='red')
).add_to(m)

# Thêm tuyến đường lên bản đồ
folium.GeoJson(route, name='Đường đi').add_to(m)

# ===== 7. Tính khoảng cách và thời gian =====
distance_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000
duration_min = route['features'][0]['properties']['segments'][0]['duration'] / 60

print(f"📏 Quãng đường: {distance_km:.2f} km")
print(f"⏱️ Dự kiến: {duration_min:.1f} phút")

# ===== 8. Lưu file HTML hiển thị bản đồ =====
m.save("duong_di.html")
print("✅ Bản đồ đã tạo xong! Mở 'duong_di.html' để xem.")
