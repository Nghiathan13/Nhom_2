import openrouteservice

# Nhập API key sau khi đăng ký tại: https://openrouteservice.org/
client = openrouteservice.Client(key= '5b3ce3597851110001cf6248b00ac6acf0fc43ec93ea37239079ce59')

# Hai điểm (kinh độ, vĩ độ)
start = (106.660172, 10.762622)  # TP.HCM
end = (106.700000, 10.780000)    # Ví dụ 1 địa điểm khác

# Gọi API để tính đường đi
route = client.directions(
    coordinates=[start, end],
    profile='driving-car',
    format='geojson'
)

# Trích xuất khoảng cách (đơn vị mét)
distance_meters = route['features'][0]['properties']['segments'][0]['distance']
duration_seconds = route['features'][0]['properties']['segments'][0]['duration']

print(f"📏 Quãng đường: {distance_meters/1000:.2f} km")
print(f"⏱️ Thời gian dự kiến: {duration_seconds/60:.1f} phút")