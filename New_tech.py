import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import locale
from sklearn.preprocessing import MinMaxScaler # <<< THÊM IMPORT NÀY

# ====================================================
# 1. Thiết lập định dạng tiền tệ Việt Nam
# ====================================================

try:
    locale.setlocale(locale.LC_ALL, 'vi_VN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'vi_VN')
    except locale.Error:
        print("Cảnh báo: Không thể đặt locale 'vi_VN'. Sử dụng định dạng mặc định.")
        def format_currency_simple(value):
            # Giữ định dạng số thập phân cho chi phí có thể nhỏ
            return "{:,.2f} VNĐ".format(value).replace(",", "X").replace(".", ",").replace("X", ".")
        format_currency = format_currency_simple
    else:
        def format_currency_locale(value):
            formatted = locale.currency(value, grouping=True, symbol=False)
            return formatted.replace(u'\xa0', u'').strip() + " VNĐ" # Bỏ khoảng trắng không ngắt
        format_currency = format_currency_locale
else:
    def format_currency_locale(value):
        formatted = locale.currency(value, grouping=True, symbol=False)
        return formatted.replace(u'\xa0', u'').strip() + " VNĐ"
    format_currency = format_currency_locale

# ====================================================
# 2. Thiết lập tham số và dữ liệu cho bài toán VRP
# =====================================================

np.random.seed(42)
torch.manual_seed(42)

# --- 2.1 THAY ĐỔI: Tham số cho bài toán VRP ---
num_nodes = 100                                            # Số lượng điểm (bao gồm kho)

demands = np.random.randint(10, 35, size=num_nodes)        # Tạo ma trận nhu cầu ngẫu nhiên
demands[0] = 0
service_times = np.random.randint(5, 20, size=num_nodes)   # Tạo ma trận thời gian phục vụ ngẫu nhiên
service_times[0] = 0
coords = np.random.randint(-125, 125, size=(num_nodes, 2)) # Tạo ma trận tọa độ ngẫu nhiên
coords[0] = [0, 0]

# --- Tính toán ma trận khoảng cách Euclid (km) ---
euclid_distance = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            euclid_distance[i][j] = np.sqrt(dx**2 + dy**2)

# --- 2.2 THAY ĐỔI: Tham số cho 2 loại xe ---
vehicle_types = {
    1: {
        'name': "Loại 1 (Nhỏ)",
        'capacity': 95, # Sức chứa
        'base_speed_avg': 40.0, # km/h
        'fuel_cost_100km': 171000.0, # VNĐ
        'base_speed_for_fuel_calc': 40.0, # km/h - tốc độ chuẩn để tính độ lệch nhiên liệu
        'load_speed_reduction_factor': 0.10, # Giảm 10% tốc độ khi đầy tải
        'driver_salary_owned': 400000, # Lương xe loại 1 nếu sở hữu
        'driver_salary_hired': 800000,  # Lương xe loại 1 nếu thuê
        'max_op_time_minutes': 10 * 60  # Thời gian hoạt động tối đa
    },
    2: {
        'name': "Loại 2 (Lớn)",
        'capacity': 110, # Sức chứa (Cao hơn)
        'base_speed_avg': 38.0, # km/h (Thấp hơn)
        'fuel_cost_100km': 173000.0, # VNĐ (Cao hơn)
        'base_speed_for_fuel_calc': 38.0, # km/h - tốc độ chuẩn để tính độ lệch nhiên liệu (Thấp hơn)
        'load_speed_reduction_factor': 0.12, # Giảm 11% tốc độ khi đầy tải (Cao hơn một chút)
        'driver_salary_owned': 420000, # Lương xe loại 2 nếu sở hữu (Cao hơn)
        'driver_salary_hired': 830000,  # Lương xe loại 2 nếu thuê (Cao hơn)
        'max_op_time_minutes': 10 * 60  # Thời gian hoạt động tối đa
    }
}

# --- 2.3 Tạo ma trận tốc độ trung bình cho 2 loại xe ---
noise_matrix_base = np.random.uniform(0.8, 1.2, size=(num_nodes, num_nodes))

base_speed_matrix_kmh_type1 = vehicle_types[1]['base_speed_avg'] * noise_matrix_base # Tạo ma trận tốc độ trung bình cho loại 1
np.fill_diagonal(base_speed_matrix_kmh_type1, 0)

base_speed_matrix_kmh_type2 = vehicle_types[2]['base_speed_avg'] * noise_matrix_base # Tạo ma trận tốc độ trung bình cho loại 2
np.fill_diagonal(base_speed_matrix_kmh_type2, 0)

# --- Ma trận thời gian khởi đầu cho loại xe 1 cho GNN ---
initial_time_matrix_minutes_type1 = np.zeros_like(base_speed_matrix_kmh_type1, dtype=float)
non_zero_speed_mask = base_speed_matrix_kmh_type1 > 0
np.divide(euclid_distance, base_speed_matrix_kmh_type1, out=initial_time_matrix_minutes_type1, where=non_zero_speed_mask)
initial_time_matrix_minutes_type1[~non_zero_speed_mask] = np.inf
initial_time_matrix_minutes_type1 *= 60
initial_time_matrix_minutes_type1 = np.round(initial_time_matrix_minutes_type1)
np.fill_diagonal(initial_time_matrix_minutes_type1, 0)

# --- Ma trận thời gian khởi đầu cho loại xe 2 cho GNN ---
initial_time_matrix_minutes_type2 = np.zeros_like(base_speed_matrix_kmh_type2, dtype=float)
non_zero_speed_mask_type2 = base_speed_matrix_kmh_type2 > 0
np.divide(euclid_distance, base_speed_matrix_kmh_type2, out=initial_time_matrix_minutes_type2, where=non_zero_speed_mask_type2)
initial_time_matrix_minutes_type2[~non_zero_speed_mask_type2] = np.inf
initial_time_matrix_minutes_type2 *= 60
initial_time_matrix_minutes_type2 = np.round(initial_time_matrix_minutes_type2)
np.fill_diagonal(initial_time_matrix_minutes_type2, 0)

# --- 2.4 THAY ĐỔI: Tham số cho bài toán VRP ---
penalty_per_unvisited_node = 1500000 # Phạt cao hơn do bài toán lớn hơn

num_owned_vehicles_type1 = 30   # Số lượng xe loại 1 sở hữu
num_hired_vehicles_type1 = 30  # Số lượng xe loại 1 thuê
num_owned_vehicles_type2 = 20   # Số lượng xe loại 2 sở hữu
num_hired_vehicles_type2 = 20  # Số lượng xe loại 2 thuê

total_owned_vehicles = num_owned_vehicles_type1 + num_owned_vehicles_type2 # Tổng số xe sở hữu
total_hired_vehicles = num_hired_vehicles_type1 + num_hired_vehicles_type2 # Tổng số xe thuê
total_vehicles_available = total_owned_vehicles + total_hired_vehicles     # Tổng số xe khả dụng

# --- 2.5 THAY ĐỔI: Tham số cho Attraction động ---
WEIGHT_TRAVEL_TIME = -0.01           # Phạt nhẹ cho mỗi phút di chuyển động (để ưu tiên điểm gần hơn)
WEIGHT_TIME_PRESSURE = -0.05         # Phạt nặng hơn nếu nút làm tăng áp lực thời gian
WEIGHT_NEAR_DEPOT_BONUS = 0.1        # Thưởng nhẹ nếu nút gần kho KHI sắp hết giờ
TIME_PRESSURE_THRESHOLD_MINUTES = 70 # Sắp hết giờ nếu thời gian còn lại < 60 phút
WEIGHT_CAPACITY_UTIL_BONUS = 0.15    # Thưởng nhẹ nếu demand gần lấp đầy capacity còn lại

# ==========================================================
# 3. Định nghĩa mô hình GNN (GNNEdgeAttr)
# ===========================================================

class GNNEdgeAttr(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim, edge_feature_dim, num_heads=4):
        super(GNNEdgeAttr, self).__init__()
        self.conv1 = TransformerConv(num_features, hidden_dim, heads=num_heads, dropout=0.3, edge_dim=edge_feature_dim)
        self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.3, edge_dim=edge_feature_dim)
        self.conv3 = TransformerConv(hidden_dim * num_heads, embedding_dim, heads=1, concat=False, dropout=0.3, edge_dim=edge_feature_dim)
        self.dropout = nn.Dropout(0.3) # Dropout tránh overfitting

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf found in input node features (x)")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        if edge_attr is not None:
             if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                 print("Warning: NaN/Inf found in input edge attributes (edge_attr)")
                 edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # print(f"Output conv1 shape: {x.shape}")
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        # print(f"Output conv2 shape: {x.shape}")
        x = self.conv3(x, edge_index, edge_attr)
        # print(f"Output conv3 shape: {x.shape}")
        return x

# =========================================================
# 4. Tạo features và Data object cho GNN
# =========================================================

print("\n--- Chuẩn bị dữ liệu cho GNN ---")

# --- 4.1 Lấy các features gốc ---
original_node_features_np = np.array([[d, s, c[0], c[1]] for d, s, c in zip(demands, service_times, coords)],
                                    dtype=np.float32)
print(f"Hình dạng features gốc (Numpy): {original_node_features_np.shape}")

# --- 4.2 Tính feature mới: Demand / Distance to Depot ---
distances_to_depot = euclid_distance[:, 0]
epsilon = 1e-6
demand_dist_ratio = np.zeros_like(demands, dtype=np.float32)
non_depot_mask = distances_to_depot > 0
non_depot_indices = np.where(non_depot_mask)[0]
demand_dist_ratio[non_depot_mask] = demands[non_depot_mask] / distances_to_depot[non_depot_mask]

zero_dist_cust_mask = (distances_to_depot <= epsilon) & (demands > 0)
if np.any(zero_dist_cust_mask):
    demand_dist_ratio[zero_dist_cust_mask] = demands[zero_dist_cust_mask] / epsilon

demand_dist_ratio = demand_dist_ratio.reshape(-1, 1)
print(f"Hình dạng feature mới (Demand/Dist): {demand_dist_ratio.shape}")
print(f"Ví dụ 5 giá trị đầu của Demand/Dist: {demand_dist_ratio[:5].flatten()}")

# --- 4.3 Kết hợp tất cả các features ---
combined_features_np = np.hstack((original_node_features_np, demand_dist_ratio))
print(f"Hình dạng features kết hợp (Numpy): {combined_features_np.shape}")

# --- 4.4 Chuẩn hóa tất cả features về thang [0, 1] ---
scaler = MinMaxScaler()
normalized_features_np = scaler.fit_transform(combined_features_np)
print(f"Đã chuẩn hóa features về thang [0, 1]. Ví dụ 5 dòng đầu:")
print(normalized_features_np[:5])

# --- 4.5 Tạo tensor node_features cuối cùng ---
node_features = torch.tensor(normalized_features_np, dtype=torch.float)
print(f"Hình dạng tensor node_features cuối cùng: {node_features.shape}")

# --- 4.6 Tạo edge_index và edge_attr ---
edge_index = []
edge_attr_list = []
large_finite_val = np.finfo(np.float32).max / 10 # Chia cho 10 để tránh Overflow

for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            edge_index.append([i, j])
            time_type1 = initial_time_matrix_minutes_type1[i][j]
            time_type2 = initial_time_matrix_minutes_type2[i][j]
            feat1 = large_finite_val if np.isinf(time_type1) else float(time_type1)
            feat2 = large_finite_val if np.isinf(time_type2) else float(time_type2)
            edge_attr_list.append([feat1, feat2])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

if not edge_attr_list:
     edge_attr = torch.empty((0, 2), dtype=torch.float)
else:
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

if edge_attr.shape[0] != edge_index.shape[1]:
     raise ValueError(f"Số lượng thuộc tính cạnh ({edge_attr.shape[0]}) không khớp số lượng cạnh ({edge_index.shape[1]})")

EXPECTED_EDGE_DIM = 2
if edge_attr.nelement() > 0 and edge_attr.shape[1] != EXPECTED_EDGE_DIM :
    raise ValueError(f"Chiều thuộc tính cạnh không khớp. Kỳ vọng {EXPECTED_EDGE_DIM}, nhận được {edge_attr.shape[1]}")

print(f"Shape of edge_attr before Data: {edge_attr.shape}")

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# --- 4.7 Cập nhật kích thước GNN và khởi tạo mô hình ---
num_features = node_features.shape[1] # <<< Tự động lấy số lượng features
print(f"Số lượng node features sau khi cập nhật: {num_features}")
hidden_dim = 128
embedding_dim = 64

edge_feature_dim = EXPECTED_EDGE_DIM if data.edge_attr is not None and data.edge_attr.nelement() > 0 else None

model = GNNEdgeAttr(num_features, hidden_dim, embedding_dim, edge_feature_dim, num_heads=4)
print(f"Khởi tạo mô hình GNN với num_features = {num_features}, edge_feature_dim = {edge_feature_dim}")

# --- 4.8 Chạy GNN để tạo embeddings ---
print("\nRunning GNN to generate embeddings...")
try:
    with torch.no_grad():
        model.eval() # Set model to evaluation mode
        embeddings = model(data)
        # Check if embeddings calculation failed (e.g., returned None due to NaNs/Infs)
        if embeddings is None:
            raise ValueError("GNN embedding calculation failed, likely due to NaN/Inf values.")
        # Additional check for NaNs/Infs in the output embeddings
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            print("Warning: NaN/Inf found in output embeddings. Replacing with 0.")
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

except Exception as e:
    print(f"Lỗi trong quá trình tính toán GNN embeddings: {e}")
    # Decide how to proceed: exit, use random embeddings, etc.
    print("Sử dụng embeddings ngẫu nhiên làm phương án dự phòng.")
    embeddings = torch.randn(num_nodes, embedding_dim) # Fallback to random embeddings

print("GNN Embeddings generated.")
print(f"Shape of embeddings: {embeddings.shape}")

# =========================================================
# 5. Hàm tính toán ĐỘNG (Tốc độ)
# =========================================================

def calculate_dynamic_speed(start_node, end_node, current_load, vehicle_params, base_speed_matrix):
    if start_node == end_node:
        return 0

    capacity = vehicle_params['capacity']
    reduction_factor = vehicle_params['load_speed_reduction_factor']
    try:
        base_speed = base_speed_matrix[start_node, end_node]
    except IndexError:
         # print(f"Warning: IndexError accessing base_speed_matrix for {start_node}->{end_node}. Using avg base.")
         base_speed = vehicle_params['base_speed_avg'] # Fallback

    if base_speed <= 0 or capacity <= 0:
        return 0

    load_ratio = min(current_load / capacity, 1.0) if capacity > 0 else 0.0
    speed_multiplier = 1.0 - (reduction_factor * load_ratio)
    dynamic_speed = base_speed * speed_multiplier
    return max(dynamic_speed, 1.0) # Tránh tốc độ quá thấp hoặc âm

# =========================================================
# 6. Hàm tính toán thời gian di chuyển ĐỘNG
# =========================================================

def calculate_dynamic_leg_time_minutes(start_node, end_node, current_load, vehicle_params, distance_matrix, base_speed_matrix):
    if start_node == end_node:
        return 0
    try:
         distance_km = distance_matrix[start_node, end_node]
    except IndexError:
         # print(f"Warning: IndexError accessing distance_matrix for {start_node}->{end_node}. Returning Inf time.")
         return np.inf

    if distance_km <= 0:
        return 0

    dynamic_speed_kmh = calculate_dynamic_speed(start_node, end_node, current_load, vehicle_params, base_speed_matrix)

    if dynamic_speed_kmh <= 0:
        # print(f"Warning: Calculated dynamic speed is <= 0 for {start_node}->{end_node}. Returning Inf time.")
        return np.inf # Trả về vô hạn nếu tốc độ <= 0

    try:
        time_hours = distance_km / dynamic_speed_kmh
        time_minutes = time_hours * 60
        # Kiểm tra NaN/Inf sau phép chia
        if not np.isfinite(time_minutes):
            # print(f"Warning: Non-finite time ({time_minutes}) calculated for {start_node}->{end_node}. Returning Inf.")
            return np.inf
        # Làm tròn về số nguyên gần nhất sau khi nhân với 60
        return round(time_minutes)
    except ZeroDivisionError:
        # print(f"Warning: ZeroDivisionError calculating time for {start_node}->{end_node}. Returning Inf time.")
        return np.inf

# ========================================================
# 7. Hàm compute_attraction
# ========================================================

def compute_attraction(current_embedding, candidate_embeddings):
    current_embedding = torch.nan_to_num(current_embedding, nan=0.0, posinf=1e6, neginf=-1e6)
    candidate_embeddings = torch.nan_to_num(candidate_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    epsilon_norm = 1e-8
    current_norm = torch.linalg.norm(current_embedding, ord=2)
    if current_norm > epsilon_norm:
       current_embedding = current_embedding / current_norm
    else:
       current_embedding.zero_() 
    
    candidate_norms = torch.linalg.norm(candidate_embeddings, ord=2, dim=1, keepdim=True)
    safe_norms = torch.clamp(candidate_norms, min=epsilon_norm)
    candidate_embeddings = candidate_embeddings / safe_norms
    candidate_embeddings[candidate_norms.squeeze() <= epsilon_norm] = 0.0
    attractions = torch.matmul(candidate_embeddings, current_embedding)
    attractions = torch.clamp(attractions, min=-1.0, max=1.0)
    return attractions

# ========================================================
# 8. Hàm build_route
# ========================================================

def build_route(embeddings, demands, service_times,
                vehicle_params, # Params của loại xe này
                visited_nodes_mask,
                distance_matrix,
                base_speed_matrix, # Ma trận tốc độ của loại xe này
                # Trọng số điều chỉnh động
                weight_travel_time=WEIGHT_TRAVEL_TIME,
                weight_time_pressure=WEIGHT_TIME_PRESSURE,
                weight_near_depot_bonus=WEIGHT_NEAR_DEPOT_BONUS,
                time_pressure_threshold=TIME_PRESSURE_THRESHOLD_MINUTES):
    """Xây dựng tuyến đường cho một loại xe, xem xét attraction động."""

    route = [0]
    current_load = 0 # Tải trọng hiện tại
    current_time = 0 # Thời gian hoạt động (phút)
    current_node = 0 # Bắt đầu từ kho (node 0)
    vehicle_capacity = vehicle_params['capacity']
    vehicle_max_time = vehicle_params['max_op_time_minutes']

    # Kiểm tra xem embeddings có chứa NaN/Inf không
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
       print("Warning: NaNs/Infs detected in embeddings tensor at start of build_route. Clamping.")
       embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6) # Replace NaNs/Infs

    # Kiểm tra node index hợp lệ trước khi truy cập embedding
    if not (0 <= current_node < embeddings.shape[0]):
        print(f"Error: current_node index {current_node} out of bounds for embeddings shape {embeddings.shape}")
        return [], 0 # Trả về tuyến rỗng nếu không thể bắt đầu

    current_embedding = embeddings[current_node]

    while True:
        # Tìm ứng viên chưa thăm
        candidate_nodes_indices = [i for i in range(1, num_nodes) if not visited_nodes_mask[i]]
        if not candidate_nodes_indices:
            break # Hết khách hàng tiềm năng

        # Lọc ứng viên theo sức chứa
        valid_candidates = []
        candidate_embeddings_list = []
        candidate_demands_list = [] # Lưu demand để tính toán sau

        for node_idx in candidate_nodes_indices:
            # Kiểm tra index hợp lệ cho demands
            if not (0 <= node_idx < len(demands)):
                # print(f"Warning: node_idx {node_idx} out of bounds for demands length {len(demands)}. Skipping candidate.")
                continue
             # Kiểm tra index hợp lệ cho embeddings
            if not (0 <= node_idx < embeddings.shape[0]):
                # print(f"Warning: node_idx {node_idx} out of bounds for embeddings shape {embeddings.shape}. Skipping candidate.")
                continue

            if current_load + demands[node_idx] <= vehicle_capacity:
                valid_candidates.append(node_idx)
                # Chỉ lấy embedding nếu index hợp lệ
                candidate_embeddings_list.append(embeddings[node_idx])
                candidate_demands_list.append(demands[node_idx])


        if not valid_candidates:
            # print("Không còn ứng viên nào phù hợp sức chứa.")
            break # Không còn ứng viên nào phù hợp sức chứa

        # --- Tính Attraction Gốc và Điều chỉnh Động ---
        if not candidate_embeddings_list: # Should not happen if valid_candidates is not empty, but check anyway
            # print("Warning: candidate_embeddings_list became empty unexpectedly.")
            break

        try:
            # Tạo tensor ngay trước khi gọi compute_attraction
            candidate_embeddings_tensor = torch.stack(candidate_embeddings_list)
            # Đảm bảo tensor đầu vào cho compute_attraction không có NaN/Inf
            if torch.isnan(candidate_embeddings_tensor).any() or torch.isinf(candidate_embeddings_tensor).any():
                print("Warning: NaNs/Infs detected in candidate_embeddings_tensor before compute_attraction. Clamping.")
                candidate_embeddings_tensor = torch.nan_to_num(candidate_embeddings_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(current_embedding).any() or torch.isinf(current_embedding).any():
                print("Warning: NaNs/Infs detected in current_embedding before compute_attraction. Clamping.")
                current_embedding = torch.nan_to_num(current_embedding, nan=0.0, posinf=1e6, neginf=-1e6)

            # Gọi hàm compute_attraction đã được cập nhật
            attractions_base = compute_attraction(current_embedding, candidate_embeddings_tensor)
            if torch.isnan(attractions_base).any():
                 print("Warning: NaNs detected in attractions_base output. Replacing with -inf.")
                 attractions_base = torch.nan_to_num(attractions_base, nan=-float('inf'))


        except RuntimeError as e:
            print(f"Error stacking candidate embeddings or computing attraction: {e}")
            print(f"Current embedding shape: {current_embedding.shape}")
            if candidate_embeddings_list:
                 shapes = [emb.shape for emb in candidate_embeddings_list]
                 print(f"Candidate shapes: {shapes}")
            break # Không thể tính attraction

        adjusted_attractions = []
        # --- Vòng lặp tính điểm điều chỉnh ---
        for i, next_node in enumerate(valid_candidates):
             # Lấy base_attraction tương ứng, cẩn thận với NaN/Inf
             if i < len(attractions_base):
                 base_attraction_score_tensor = attractions_base[i]
                 if torch.isnan(base_attraction_score_tensor) or torch.isinf(base_attraction_score_tensor):
                      base_attraction_score = -float('inf') # Gán giá trị rất thấp nếu có vấn đề
                 else:
                      base_attraction_score = base_attraction_score_tensor.item()
             else:
                 print(f"Warning: Index {i} out of bounds for attractions_base (len={len(attractions_base)}). Setting base score to -inf.")
                 base_attraction_score = -float('inf')


             dynamic_adjustment = 0.0

             # --- 1. Phạt thời gian di chuyển động ---
             travel_time_to_next = calculate_dynamic_leg_time_minutes(
                 current_node, next_node, current_load, vehicle_params,
                 distance_matrix, base_speed_matrix
             )

             if np.isinf(travel_time_to_next):
                 dynamic_adjustment = -float('inf') # Rất tiêu cực nếu không thể đi đến
                 final_score = -float('inf')
             else:
                 dynamic_adjustment += weight_travel_time * travel_time_to_next

                 # --- 2. Phạt/Thưởng dựa trên áp lực thời gian ---
                 # Chỉ tính toán nếu có thể đến được nút tiếp theo
                 # Check indices for service_times and demands
                 if not (0 <= next_node < len(service_times)) or not (0 <= next_node < len(demands)):
                    # print(f"Warning: next_node {next_node} out of bounds for service_times/demands. Skipping time pressure calc.")
                    # Nếu không tính được, coi như điểm này rủi ro cao về thời gian
                    dynamic_adjustment += -1e4 # Phạt nặng nếu index lỗi
                 else:
                    service_time_at_next = service_times[next_node]
                    load_after_next = current_load + demands[next_node]
                    return_time_from_next = calculate_dynamic_leg_time_minutes(
                        next_node, 0, load_after_next, vehicle_params, distance_matrix, base_speed_matrix
                    )

                    if np.isinf(return_time_from_next):
                        # Nếu không thể quay về từ nút tiếp theo, điểm này rất tệ
                        dynamic_adjustment = -float('inf')
                    else:
                        projected_end_time = current_time + travel_time_to_next + service_time_at_next + return_time_from_next
                        remaining_time_at_end = vehicle_max_time - projected_end_time

                        if remaining_time_at_end < time_pressure_threshold:
                            # Phạt nếu còn ít thời gian
                            dynamic_adjustment += weight_time_pressure * (time_pressure_threshold - remaining_time_at_end)

                            # Chỉ thưởng gần kho khi thời gian gấp
                            # Kiểm tra index của next_node trước khi truy cập distance_matrix
                            if 0 <= next_node < distance_matrix.shape[0]:
                                dist_to_depot = distance_matrix[next_node, 0]
                                # Giới hạn bonus để không quá lớn
                                depot_bonus = max(0, weight_near_depot_bonus * (100 - dist_to_depot)) # VD: gần hơn 100km
                                dynamic_adjustment += depot_bonus
                            # else:
                                # print(f"Warning: next_node {next_node} out of bounds for distance_matrix in depot bonus calc.")

                 # --- 3. Thưởng hiệu quả tải trọng ---
                 # Chỉ tính toán nếu không bị phạt vô hạn ở các bước trước
                 if not np.isinf(dynamic_adjustment):
                      remaining_capacity = vehicle_capacity - current_load
                      # Lấy demand từ list đã lưu, đảm bảo index i hợp lệ
                      if i < len(candidate_demands_list):
                           demand_at_node = candidate_demands_list[i]
                           # Chỉ thưởng nếu có capacity còn lại và demand tận dụng tốt capacity đó
                           if remaining_capacity > 0 and demand_at_node > 0:
                                util_ratio = demand_at_node / remaining_capacity
                                if util_ratio > 0.8: # Ví dụ: lấp đầy > 80% capacity *còn lại*
                                   # Bonus tỉ lệ với base_attraction để nó không lấn át quá nhiều
                                   # Kiểm tra base_attraction_score trước khi nhân
                                   if np.isfinite(base_attraction_score):
                                      dynamic_adjustment += WEIGHT_CAPACITY_UTIL_BONUS * abs(base_attraction_score) # Dùng abs phòng trường hợp base<0
                      else:
                           print(f"Warning: Index {i} out of bounds for candidate_demands_list.")


             # Tính điểm cuối cùng, đảm bảo không phải là NaN/Inf trừ khi cố ý
             if np.isinf(dynamic_adjustment):
                 final_score = dynamic_adjustment # Giữ -inf
             elif np.isnan(base_attraction_score) or np.isinf(base_attraction_score):
                 # Nếu base score không hợp lệ nhưng adjustment thì có, ưu tiên adjustment?
                 # Hoặc đơn giản là bỏ qua nếu base score tệ? --> Bỏ qua có vẻ an toàn hơn.
                 final_score = -float('inf')
             else:
                 final_score = base_attraction_score + dynamic_adjustment
                 if not np.isfinite(final_score): # Kiểm tra lần cuối
                     final_score = -float('inf')

             adjusted_attractions.append(final_score)
             # Debug print (optional):
             # print(f"  Node {next_node}: Base={base_attraction_score:.2f}, Adj={dynamic_adjustment:.2f}, Final={final_score:.2f}")

        # Sắp xếp ứng viên dựa trên điểm đã điều chỉnh
        # Lấy index trong list adjusted_attractions (tương ứng với valid_candidates)
        try:
            # Chỉ sắp xếp những điểm có điểm số hữu hạn
            finite_scores_indices = [idx for idx, score in enumerate(adjusted_attractions) if np.isfinite(score)]
            if not finite_scores_indices: # Nếu không có điểm nào hữu hạn
                 # print("Tất cả điểm điều chỉnh là -inf, không thể sắp xếp.")
                 indices_sorted_by_adjusted = []
            else:
                # Tạo tuple (điểm, chỉ số gốc âm) CHỈ cho các điểm hữu hạn
                sort_keys = [(-adjusted_attractions[idx], idx) for idx in finite_scores_indices]
                # Sắp xếp key của các điểm hữu hạn
                sorted_finite_key_indices = sorted(finite_scores_indices, key=lambda k: (-adjusted_attractions[k], k))
                # sorted_key_indices = sorted(range(len(sort_keys)), key=lambda k: sort_keys[k])

                # Lấy ra indices tương ứng với list valid_candidates ban đầu
                indices_sorted_by_adjusted = sorted_finite_key_indices # Đây là indices trong valid_candidates

        except Exception as e:
            print(f"Error sorting adjusted attractions: {e}. Falling back to base attraction sort.")
            # Phương án dự phòng: sắp xếp theo attraction gốc nếu có lỗi và base hợp lệ
            if 'attractions_base' in locals() and attractions_base is not None and torch.isfinite(attractions_base).all():
                 indices_sorted_by_adjusted = torch.argsort(attractions_base, descending=True).tolist()
            else:
                 indices_sorted_by_adjusted = [] # Không thể sắp xếp dự phòng

        # --- Vòng lặp kiểm tra và thêm node theo thứ tự mới ---
        node_added_in_step = False
        # Chỉ lặp qua các index đã được sắp xếp thành công
        for idx_in_valid_candidates in indices_sorted_by_adjusted:
            # Lấy lại next_node từ list gốc valid_candidates dùng chỉ số đã sắp xếp
            # Cần kiểm tra idx_in_valid_candidates có hợp lệ không
             if not (0 <= idx_in_valid_candidates < len(valid_candidates)):
                 print(f"Warning: Sorted index {idx_in_valid_candidates} out of bounds for valid_candidates.")
                 continue

             next_node = valid_candidates[idx_in_valid_candidates]


             # --- KIỂM TRA LẠI KHẢ THI CUỐI CÙNG ---
             # Đảm bảo indices hợp lệ trước khi tính toán
             if not (0 <= current_node < num_nodes and 0 <= next_node < num_nodes):
                # print(f"Warning: Invalid node indices c={current_node} or n={next_node} before final check. Skipping.")
                continue

             # 1. Tính thời gian đến nút tiếp theo
             travel_time_to_next = calculate_dynamic_leg_time_minutes(
                 current_node, next_node, current_load, vehicle_params,
                 distance_matrix, base_speed_matrix
             )
             if np.isinf(travel_time_to_next):
                 # print(f"   - Skip Node {next_node}: Cannot reach (time=inf).")
                 continue # Không thể đi

             # 2. Tính thời gian phục vụ và tải trọng mới
             # Ensure valid index for service_times and demands before access
             if not (0 <= next_node < len(service_times)) or not (0 <= next_node < len(demands)):
                # print(f"Warning: next_node {next_node} out of bounds during final feasibility check. Skipping.")
                continue
             service_time_at_next = service_times[next_node]
             # Tải trọng mới đã được kiểm tra ở bước lọc đầu tiên
             load_after_next = current_load + demands[next_node]

             # 3. Tính thời gian quay về kho TỪ nút tiếp theo
             return_time_from_next = calculate_dynamic_leg_time_minutes(
                 next_node, 0, load_after_next, vehicle_params,
                 distance_matrix, base_speed_matrix
             )
             if np.isinf(return_time_from_next):
                 # print(f"   - Skip Node {next_node}: Cannot return to depot from it (time=inf).")
                 continue # Không thể quay về kho từ đó

             # 4. Kiểm tra TỔNG thời gian dự kiến
             projected_total_time = current_time + travel_time_to_next + service_time_at_next + return_time_from_next

             if projected_total_time > vehicle_max_time:
                # print(f"   - Skip Node {next_node}: Exceeds max time ({projected_total_time:.0f} > {vehicle_max_time}).")
                continue # Vi phạm thời gian -> thử ứng viên tiếp theo

             # --- THÊM NODE NẾU TẤT CẢ HỢP LỆ ---
             route.append(next_node)
             visited_nodes_mask[next_node] = True # Đánh dấu đã thăm trong MASK GỐC
             current_load = load_after_next
             current_time += travel_time_to_next + service_time_at_next
             current_node = next_node # Di chuyển đến node mới

             # --- Cập nhật embedding hiện tại cho bước lặp sau ---
             if not (0 <= current_node < embeddings.shape[0]):
                print(f"Error: current_node index {current_node} became out of bounds after adding. Stopping route construction.")
                node_added_in_step = False # Đánh dấu là không thêm được nữa để thoát vòng ngoài
                break
             current_embedding = embeddings[current_node] # Cập nhật embedding cho nút hiện tại mới

             node_added_in_step = True
             # print(f"      Node {next_node} added to route. Load={current_load:.1f}, Time={current_time:.0f}") # Optional debug
             break # Đã thêm node thành công -> chuyển sang bước tìm node tiếp theo từ current_node mới

        if not node_added_in_step:
            # print("      No suitable node found in this step, breaking route construction.")
            break # Không thêm được node nào phù hợp nữa

    # --- Kết thúc tuyến, quay về kho (nếu cần) ---
    if len(route) > 1: # Nếu tuyến có ít nhất một khách hàng
        # Check if current_node is valid before final calculation
        if not (0 <= current_node < num_nodes):
            print(f"Error: current_node {current_node} invalid before final return leg calc. Route: {route}")
            return route, current_time # Trả về như hiện tại, không thêm kho

        last_leg_time = calculate_dynamic_leg_time_minutes(
            current_node, 0, current_load, vehicle_params,
            distance_matrix, base_speed_matrix
        )

        if np.isinf(last_leg_time):
            print(f"Warning: Cannot calculate return time from {current_node} for route {route}. Returning route as is.")
            return route, current_time # Không thể quay về

        # Kiểm tra lần cuối ràng buộc thời gian KHI quay về
        if current_time + last_leg_time > vehicle_max_time:
            print(f"Warning: Returning to depot from {current_node} exceeds max time ({current_time + last_leg_time:.0f} > {vehicle_max_time}). Route: {route}. Returning route as is.")
            return route, current_time # Quay về sẽ vi phạm thời gian
        else:
            route.append(0) # Thêm kho vào cuối
            current_time += last_leg_time
            return route, current_time # Trả về tuyến hoàn chỉnh và tổng thời gian
    else:
        # Tuyến chỉ có [0] hoặc []
        return [], 0

# ========================================================
# 9. Hàm tính chi phí nhiên liệu
# ========================================================

def calculate_dynamic_leg_fuel_cost(start_node, end_node, current_load,
                                    vehicle_params, # << Params của loại xe này
                                    distance_matrix, base_speed_matrix): # << Ma trận tốc độ của loại xe này
    """Tính chi phí nhiên liệu động cho một chặng VÀ LOẠI XE."""
    if start_node == end_node: return 0.0 # Trả về float
    try: distance_km = distance_matrix[start_node, end_node]
    except IndexError: return 0.0

    if distance_km <= 0: return 0.0

    base_cost_per_100km = vehicle_params['fuel_cost_100km']
    base_speed_for_fuel = vehicle_params['base_speed_for_fuel_calc']

    # Lấy tốc độ động (nên gọi lại hàm hoặc đảm bảo logic đồng nhất)
    dynamic_speed_kmh = calculate_dynamic_speed(start_node, end_node, current_load, vehicle_params, base_speed_matrix)

    if dynamic_speed_kmh <= 0: return 0.0 # Hoặc giá trị phạt rất lớn nếu không thể di chuyển?

    # --- Simple Fuel Cost Multiplier ---
    if base_speed_for_fuel <= 0:
        cost_multiplier = 1.0
    else:
        speed_diff_ratio = abs(dynamic_speed_kmh - base_speed_for_fuel) / base_speed_for_fuel
        # Giảm nhẹ ảnh hưởng của tốc độ lệch, tránh tăng chi phí quá đà
        cost_multiplier = 1.0 + (speed_diff_ratio * 0.25) # VD: lệch 20% -> tăng 5% chi phí
        cost_multiplier = max(1.0, cost_multiplier) # Chi phí không bao giờ giảm do tốc độ lệch

    base_cost_per_km = base_cost_per_100km / 100.0
    leg_cost = distance_km * base_cost_per_km * cost_multiplier

    # Đảm bảo kết quả là số hữu hạn
    return leg_cost if np.isfinite(leg_cost) else np.finfo(np.float32).max / 10.0

# ========================================================
# 10. Xây dựng các tuyến đường với chiến lược phân bổ xe
# ========================================================

visited = [False] * num_nodes
visited[0] = True # Kho luôn được coi là "đã thăm" ban đầu

routes_info = []
num_vehicles_used_total = 0
available_counts = {
    'type1_owned': num_owned_vehicles_type1,
    'type1_hired': num_hired_vehicles_type1,
    'type2_owned': num_owned_vehicles_type2,
    'type2_hired': num_hired_vehicles_type2
}
actual_used_counts = {'type1_owned': 0, 'type1_hired': 0, 'type2_owned': 0, 'type2_hired': 0}

print("\n--- Bắt đầu Xây dựng Tuyến đường ---")
print(f"Trọng số điều chỉnh: Time Cost = {WEIGHT_TRAVEL_TIME}, Time Pressure = {WEIGHT_TIME_PRESSURE}, Depot Bonus = {WEIGHT_NEAR_DEPOT_BONUS}, Capacity Bonus = {WEIGHT_CAPACITY_UTIL_BONUS}"),
print("Cấu hình Xe:")
for v_id, v_params in vehicle_types.items():
    salary_o = v_params['driver_salary_owned']
    salary_h = v_params['driver_salary_hired']
    print(f"  Loại {v_id} ({v_params['name']}): "
          f"Cap={v_params['capacity']}kg, "
          f"MaxTime={v_params['max_op_time_minutes']}m, "
          f"Cost(O/H) = {format_currency(salary_o)} / {format_currency(salary_h)}")

# Vòng lặp chính để huy động xe
while True:
    unvisited_customer_exists = any(not visited[j] for j in range(1, num_nodes))
    if not unvisited_customer_exists:
        print(f"\nThông báo: Tất cả các điểm khách hàng đã được phục vụ. Dừng huy động xe.")
        break

    available_options = []
    if available_counts['type1_owned'] > 0: available_options.append({'type': 1, 'status': 'owned'})
    if available_counts['type2_owned'] > 0: available_options.append({'type': 2, 'status': 'owned'})
    if available_counts['type1_hired'] > 0: available_options.append({'type': 1, 'status': 'hired'})
    if available_counts['type2_hired'] > 0: available_options.append({'type': 2, 'status': 'hired'})

    if not available_options:
        print(f"\nThông báo: Đã hết loại xe phù hợp HOẶC không còn xe nào trong đội. Dừng huy động.")
        break

    print(f"\nĐang tìm tuyến cho Xe thứ {num_vehicles_used_total + 1}. Lựa chọn còn: "
            f"1(O):{available_counts['type1_owned']}, 2(O):{available_counts['type2_owned']}, "
            f"1(H):{available_counts['type1_hired']}, 2(H):{available_counts['type2_hired']}")

    simulation_results = []
    # --- Vòng lặp thử nghiệm các loại xe ---
    for option in available_options:
        veh_type_id = option['type']
        status = option['status']
        veh_params = vehicle_types[veh_type_id]
        vehicle_capacity = veh_params['capacity']
        base_speed_matrix = base_speed_matrix_kmh_type1 if veh_type_id == 1 else base_speed_matrix_kmh_type2

        # print(f"  -> Thử nghiệm: Loại {veh_type_id} ({status})...") # Ít chi tiết hơn
        visited_copy = visited[:] # Create a copy for simulation
        try:
            # Sử dụng embeddings mới đã chuẩn hóa khi gọi build_route
            sim_route, sim_op_time = build_route(
                embeddings, demands, service_times,
                veh_params,
                visited_copy, # Sử dụng MASK giả lập
                euclid_distance,
                base_speed_matrix,
                # Các trọng số đã định nghĩa ở global scope sẽ được dùng mặc định
                weight_travel_time=WEIGHT_TRAVEL_TIME,
                weight_time_pressure=WEIGHT_TIME_PRESSURE,
                weight_near_depot_bonus=WEIGHT_NEAR_DEPOT_BONUS,
                time_pressure_threshold=TIME_PRESSURE_THRESHOLD_MINUTES
                # WEIGHT_CAPACITY_UTIL_BONUS được dùng bên trong build_route
            )
        except Exception as build_err:
             print(f"    Lỗi khi thử nghiệm build_route cho Loại {veh_type_id} ({status}): {build_err}")
             sim_route = [] # Đảm bảo không xử lý tuyến lỗi

        # --- Đánh giá tuyến thử nghiệm ---
        if sim_route and len(sim_route) > 2: # Chỉ xét tuyến có ít nhất 1 KH (0 -> KH -> 0)
            sim_total_demand = 0
            sim_fuel_cost = 0
            current_sim_load = 0
            sim_distance = 0 # Track distance

            # Tính toán lại demand và fuel cost chính xác cho tuyến giả lập
            for j in range(len(sim_route) - 1):
                start_node = sim_route[j]
                end_node = sim_route[j+1]
                if not (0 <= start_node < num_nodes and 0 <= end_node < num_nodes):
                    print(f"Warning (Sim Metric): Invalid indices s={start_node}, e={end_node} in route {sim_route}. Skipping leg.")
                    continue

                leg_dist = euclid_distance[start_node, end_node]
                sim_distance += leg_dist
                # Sử dụng hàm tính fuel cost động
                leg_fuel_cost = calculate_dynamic_leg_fuel_cost(
                    start_node, end_node, current_sim_load, veh_params,
                    euclid_distance, base_speed_matrix
                 )
                sim_fuel_cost += leg_fuel_cost

                # Cập nhật tải trọng CHO CHẶNG TIẾP THEO của simulation
                if end_node != 0: # Nếu đích là khách hàng
                    if not (0 <= end_node < len(demands)):
                         print(f"Warning (Sim Metric): Invalid end_node {end_node} for demands lookup.")
                         continue
                    demand_at_node = demands[end_node]
                    sim_total_demand += demand_at_node
                    current_sim_load += demand_at_node # << QUAN TRỌNG: Cập nhật load

            # Chi phí cố định (lương)
            sim_driver_salary = veh_params[f'driver_salary_{status}']
            sim_total_cost = sim_fuel_cost + sim_driver_salary # Chi phí tiền tệ tổng cộng

            # Metrics đánh giá
            # Handle division by zero if cost is zero (shouldn't happen if salary>0)
            sim_demand_per_cost = (sim_total_demand * 1000 / sim_total_cost) if sim_total_cost > 0 else 0.0
            sim_utilization = (sim_total_demand / vehicle_capacity) if vehicle_capacity > 0 else 0.0
            time_utilization = (sim_op_time / veh_params['max_op_time_minutes']) if veh_params['max_op_time_minutes'] > 0 else 0.0

            # Tính số nút mới được phục vụ BỞI TUYẾN GIẢ LẬP NÀY
            served_new_nodes_count = sum(1 for node in sim_route if node != 0 and not visited[node]) # So với MASK GỐC 'visited'

            # Score tổng hợp - Có thể cần tinh chỉnh trọng số
            util_weight = 0.25  # Trọng số cho hiệu quả tải trọng (%)
            demand_cost_weight = 0.30 # Trọng số cho hiệu quả chi phí (g/VND)
            new_node_weight = 0.35   # Trọng số cho số KH mới
            time_util_weight = 0.10 # Trọng số cho hiệu quả thời gian (%)

            normalized_dpc = min(sim_demand_per_cost / 100.0, 1.0) if sim_demand_per_cost > 0 else 0.0

            sim_score = (
                (new_node_weight * served_new_nodes_count) +       # Phần thưởng lớn cho mỗi KH mới
                (demand_cost_weight * normalized_dpc * 10 ) +      # Điểm cho hiệu quả chi phí (nhân 10 để tăng ảnh hưởng)
                (util_weight * sim_utilization) +                  # Điểm cho hiệu quả tải trọng
                (time_util_weight * time_utilization)              # Điểm cho hiệu quả thời gian
            )

            # Chỉ thêm vào kết quả nếu tuyến này phục vụ ít nhất 1 KH mới
            # Hoặc nếu không còn KH nào chưa thăm, thì chọn tuyến tốt nhất có thể
            if served_new_nodes_count > 0 :
                 print(f"     + Thử nghiệm OK (Loại {veh_type_id}/{status}): {served_new_nodes_count} KH mới, "
                       f"Demand: {sim_total_demand:.0f}kg ({sim_utilization:.1%}), "
                       f"Cost: {format_currency(sim_total_cost)}, D/C: {sim_demand_per_cost:.2f} g/VNĐ, "
                       f"Dist: {sim_distance:.1f}km, Time: {sim_op_time:.0f}m ({time_utilization:.1%}), "
                       f"Score: {sim_score:.2f}") # Dùng sim_score mới

                 simulation_results.append({
                     'route': sim_route,
                     'op_time': sim_op_time,
                     'nodes_served': len([node for node in sim_route if node != 0]),
                     'served_new': served_new_nodes_count, # Số nút mới tuyến này sẽ thăm
                     'sim_total_demand': sim_total_demand,
                     'sim_fuel_cost': sim_fuel_cost,
                     'sim_driver_salary': sim_driver_salary,
                     'sim_total_cost': sim_total_cost,
                     'sim_demand_per_cost': sim_demand_per_cost,
                     'sim_utilization': sim_utilization,
                     'sim_score': sim_score, # Sử dụng score mới
                     'type': veh_type_id,
                     'status': status,
                     'capacity': vehicle_capacity,
                     'distance': sim_distance,
                     'time_utilization': time_utilization
                 })
            # else:
                 # print(f"     - Thử nghiệm Loại {veh_type_id}/{status}: Không phục vụ KH mới. Bỏ qua.")
        # else:
            # print(f"     - Thử nghiệm Loại {veh_type_id}/{status}: Không tạo được tuyến hợp lệ hoặc quá ngắn.")


    # --- Chọn tuyến đường tối ưu ---
    if not simulation_results:
        # Kiểm tra lại lần cuối xem còn khách nào không trước khi dừng hẳn
        if any(not visited[j] for j in range(1, num_nodes)):
            print(f"\nCảnh báo: Không thể tạo được tuyến đường hữu ích nào bằng xe còn lại dù vẫn còn KH. Dừng huy động.")
        else:
            # Trường hợp này không nên xảy ra nếu vòng lặp ngoài đã kiểm tra
             print(f"\nThông báo: Không còn kết quả thử nghiệm và không còn KH chưa thăm.")
        break # Dừng vòng lặp huy động xe

    # Sắp xếp theo điểm số tổng hợp mới, sau đó các yếu tố phụ
    simulation_results.sort(key=lambda x: (
        -x['sim_score'],                # 1. Maximize điểm tổng hợp mới
        -x['served_new'],               # 2. Maximize số KH mới (vẫn quan trọng)
        -x['sim_demand_per_cost'],      # 3. Maximize hiệu quả chi phí (phụ)
        -x['sim_utilization'],          # 4. Maximize hiệu quả tải trọng (phụ)
         x['sim_total_cost'],           # 5. Minimize chi phí (phụ)
         0 if x['status'] == 'owned' else 1, # 6. Ưu tiên xe 'owned'
    ))

    best_result = simulation_results[0]

    # Kiểm tra cuối cùng xem lựa chọn tốt nhất có thực sự phục vụ gì không
    # (Mặc dù logic lọc simulation_results đã cố gắng đảm bảo served_new > 0)
    if best_result['served_new'] <= 0:
        if any(not visited[j] for j in range(1, num_nodes)):
            print(f"\nCảnh báo: Lựa chọn tốt nhất (Loại {best_result['type']}/{best_result['status']}) không phục vụ KH mới nào dù vẫn còn KH. Dừng huy động.")
        else:
            print(f"\nThông báo: Lựa chọn tốt nhất không phục vụ KH mới, và cũng không còn KH chưa phục vụ. Dừng.")
        break

    print(f"  => Lựa chọn Tốt nhất: Loại {best_result['type']} ({best_result['status']}) "
          f"- Score: {best_result['sim_score']:.2f}, "
          f"KH mới: {best_result['served_new']}, "
          f"Demand: {best_result['sim_total_demand']:.0f}kg ({best_result['sim_utilization']:.1%}), "
          f"Cost: {format_currency(best_result['sim_total_cost'])}")

    # --- Cam kết sử dụng tuyến tốt nhất ---
    best_type = best_result['type']
    best_status = best_result['status']

    # Giảm số lượng xe khả dụng
    vehicle_key = f'type{best_type}_{best_status}'
    if available_counts[vehicle_key] > 0:
        available_counts[vehicle_key] -= 1
    else:
        print(f"!!! Lỗi nghiêm trọng: Cố gắng dùng xe {vehicle_key} nhưng số lượng khả dụng là 0. Dừng lại. !!!")
        break # Dừng nếu có sự không nhất quán nghiêm trọng

    # Cập nhật MASK visited THỰC TẾ dựa trên tuyến đã chọn
    final_route = best_result['route']
    newly_visited_count_actual = 0
    for node_idx in final_route:
        if node_idx != 0: # Chỉ cập nhật cho khách hàng
             if not (0 <= node_idx < len(visited)):
                  print(f"Warning (Commit): node_idx {node_idx} out of bounds for visited mask {len(visited)}. Skipping update.")
                  continue
             if not visited[node_idx]: # Chỉ đánh dấu và đếm nếu chưa thăm
                visited[node_idx] = True
                newly_visited_count_actual += 1

    if newly_visited_count_actual != best_result['served_new']:
         print(f"Warning: Số nút mới thực tế ({newly_visited_count_actual}) khác số nút mới ước tính ({best_result['served_new']}) cho tuyến {num_vehicles_used_total + 1}.")

    # Lưu thông tin tuyến cuối cùng
    routes_info.append({
        'route': final_route,
        'type': best_type,
        'status': best_status,
        'op_time': best_result['op_time'],
        # Lưu lại các metrics từ best_result để tiện so sánh/phân tích
        'final_demand': best_result['sim_total_demand'],
        'final_fuel_cost_est': best_result['sim_fuel_cost'],
        'final_salary_cost': best_result['sim_driver_salary'],
        'final_total_cost_est': best_result['sim_total_cost'],
        'final_utilization': best_result['sim_utilization'],
        'capacity': best_result['capacity'],
        'distance_est': best_result['distance'], # Đổi tên cho rõ là ước tính
        'time_utilization': best_result['time_utilization'],
        'num_customers': best_result['nodes_served'] # Số KH trên tuyến này
    })

    num_vehicles_used_total += 1
    actual_used_counts[f'type{best_type}_{best_status}'] += 1

    print(f"  -> Xe {num_vehicles_used_total} (Loại {best_type} - {best_status.capitalize()}) được huy động.")
    route_str = str(final_route)
    if len(route_str) > 80: route_str = route_str[:38] + "..." + route_str[-38:]
    print(f"     Tuyến ({best_result['nodes_served']} KH): {route_str}")
    print(f"     Thời gian: {best_result['op_time']:.0f} phút ({best_result['time_utilization']:.1%})")
    print(f"     Nhu cầu/Cap: {best_result['sim_total_demand']:.0f}kg / {best_result['capacity']}kg ({best_result['sim_utilization']:.1%})")
    print(f"     Chi phí Ước tính: {format_currency(best_result['sim_total_cost'])}")
    print(f"     Quãng đường Ước tính: {best_result['distance']:.2f} km")

# --- Kết thúc vòng lặp xây dựng tuyến ---

# Kiểm tra trạng thái cuối cùng
if not any(not visited[j] for j in range(1, num_nodes)):
    print(f"\nTất cả {num_nodes-1} điểm khách hàng đã được phục vụ!")
elif any(not visited[j] for j in range(1, num_nodes)):
    total_used = sum(actual_used_counts.values())
    total_initial_available = num_owned_vehicles_type1 + num_hired_vehicles_type1 + num_owned_vehicles_type2 + num_hired_vehicles_type2

    if total_used >= total_initial_available :
        print(f"\nCảnh báo: Đã sử dụng hết TOÀN BỘ xe ({total_used}/{total_initial_available} xe) nhưng vẫn còn điểm chưa phục vụ.")
    else:
        print(f"\nCảnh báo: Dừng huy động xe (đã dùng {total_used}/{total_initial_available} xe) do không tạo được tuyến hiệu quả nữa, vẫn còn điểm chưa phục vụ.")

# =========================================================
# 11. Phân tích Kết quả và Tổng Chi Phí
# =========================================================

print("\n--- Phân tích Kết quả và Tổng Chi Phí (VNĐ) ---")

total_final_fuel_cost = 0.0
total_final_driver_salary = 0.0
total_final_distance = 0.0
final_metrics_per_route = [] # Lưu metric thực tế của mỗi tuyến

# Tính toán lại chi phí và quãng đường THỰC TẾ cho từng tuyến đã chốt
for i, info in enumerate(routes_info):
    route = info['route']
    veh_type_id = info['type']
    status = info['status']
    veh_params = vehicle_types[veh_type_id]
    base_speed_matrix = base_speed_matrix_kmh_type1 if veh_type_id == 1 else base_speed_matrix_kmh_type2
    final_salary = veh_params[f'driver_salary_{status}'] # Lương thực tế

    route_final_fuel = 0.0
    route_final_dist = 0.0
    current_load_final = 0.0 # Reset load for final calculation

    # Lặp qua các chặng của tuyến để tính chi phí/quãng đường thực tế
    if len(route) > 1: # Chỉ tính nếu tuyến có di chuyển
        for j in range(len(route) - 1):
            start_node = route[j]
            end_node = route[j+1]

            if not (0 <= start_node < num_nodes and 0 <= end_node < num_nodes):
                print(f"Error (Final Cost): Invalid indices s={start_node}, e={end_node} in final route {i+1}. Skipping leg cost.")
                continue

            # 1. Tính quãng đường thực tế của chặng
            leg_dist = euclid_distance[start_node, end_node]
            route_final_dist += leg_dist

            # 2. Tính chi phí nhiên liệu thực tế của chặng
            leg_fuel = calculate_dynamic_leg_fuel_cost(
                start_node, end_node, current_load_final, veh_params,
                euclid_distance, base_speed_matrix
            )
            route_final_fuel += leg_fuel

            # 3. Cập nhật tải trọng cho việc tính chi phí CHẶNG TIẾP THEO
            if end_node != 0:
                if not (0 <= end_node < len(demands)):
                    print(f"Warning (Final Cost): Invalid node {end_node} for demand lookup in route {i+1}.")
                    continue
                current_load_final += demands[end_node]

    # Tổng hợp chi phí và quãng đường thực tế của tuyến
    total_final_fuel_cost += route_final_fuel
    total_final_driver_salary += final_salary
    total_final_distance += route_final_dist
    route_final_total_cost = route_final_fuel + final_salary

    # Lưu metrics thực tế
    final_metrics_per_route.append({
        'route_index': i + 1,
        'final_distance_km': route_final_dist,
        'final_fuel_cost': route_final_fuel,
        'final_driver_salary': final_salary,
        'final_total_cost': route_final_total_cost
    })

    # --- So sánh chi phí thực tế và ước tính (từ simulation) ---
    # Lấy giá trị ước tính đã lưu trong routes_info
    est_total_cost = info.get('final_total_cost_est')
    est_dist = info.get('distance_est')

    # Optional: In ra so sánh nếu khác biệt lớn
    cost_diff = abs(route_final_total_cost - est_total_cost) if est_total_cost is not None else 0
    dist_diff = abs(route_final_dist - est_dist) if est_dist is not None else 0
    # if cost_diff > 1000 or dist_diff > 0.5: # Ngưỡng nhỏ
    #     print(f"    * Route {i+1}: Est/Final Cost: {format_currency(est_total_cost or 0)}/{format_currency(route_final_total_cost)} (Diff: {format_currency(cost_diff)})")
    #     print(f"    * Route {i+1}: Est/Final Dist: {est_dist or 0:.2f}km/{route_final_dist:.2f}km (Diff: {dist_diff:.2f}km)")


# Tính phạt cho các điểm chưa phục vụ
unvisited_nodes_list = [i for i in range(1, num_nodes) if not visited[i]]
num_unvisited = len(unvisited_nodes_list)
total_unvisited_penalty = num_unvisited * penalty_per_unvisited_node

# Tổng chi phí mục tiêu cuối cùng (dựa trên tính toán thực tế)
total_final_monetary_cost = total_final_fuel_cost + total_final_driver_salary + total_unvisited_penalty

# --- In Kết quả Cuối cùng ---
print(f"\nTổng số xe thực tế đã sử dụng: {num_vehicles_used_total} / {total_vehicles_available} khả dụng")
print(f"  - Loại 1: {actual_used_counts['type1_owned']} Sở hữu ({num_owned_vehicles_type1} dispo), {actual_used_counts['type1_hired']} Thuê ({num_hired_vehicles_type1} dispo)")
print(f"  - Loại 2: {actual_used_counts['type2_owned']} Sở hữu ({num_owned_vehicles_type2} dispo), {actual_used_counts['type2_hired']} Thuê ({num_hired_vehicles_type2} dispo)")
print(f"Tổng quãng đường di chuyển (Thực tế): {total_final_distance:.2f} km")


print("\nChi tiết các tuyến đường và chi phí (Thực tế):")
for i, info in enumerate(routes_info):
    # Lấy thông tin cơ bản từ routes_info
    route_str_details = str(info['route'])
    if len(route_str_details) > 80: route_str_details = route_str_details[:38] + "..." + route_str_details[-38:]
    veh_params = vehicle_types[info['type']]
    status_str = info['status'].capitalize()
    num_cust_route = info.get('num_customers', 'N/A')

    # Lấy chi phí/quãng đường thực tế từ final_metrics_per_route
    route_metrics = final_metrics_per_route[i] # Giả sử index khớp

    print(f"  Tuyến {i+1} (Xe {i+1} - {veh_params['name']} - {status_str}, {num_cust_route} KH): {route_str_details}")
    # In quãng đường ƯỚC TÍNH (từ build_route) và THỰC TẾ (tính lại)
    print(f"    Quãng đường (Ước tính/Thực tế): {info.get('distance_est', -1):.2f} km / {route_metrics['final_distance_km']:.2f} km")
    print(f"    Thời gian hoạt động (Ước tính): {info.get('op_time', -1):.0f} phút (Max: {veh_params['max_op_time_minutes']}, Util: {info.get('time_utilization', 0):.1%})")
    # Lấy demand, util, capacity từ info
    route_demand = info.get('final_demand', 0)
    route_util = info.get('final_utilization', 0)
    route_capacity = info.get('capacity', 0)
    print(f"    Nhu cầu / Capacity: {route_demand:.0f} kg / {route_capacity} kg ({route_util:.1%} Util)")
    # In chi phí THỰC TẾ đã tính lại
    print(f"    Chi phí nhiên liệu (Thực tế)  : {format_currency(route_metrics['final_fuel_cost'])}")
    print(f"    Chi phí lương tài xế (Thực tế): {format_currency(route_metrics['final_driver_salary'])}")
    print(f"    Tổng chi phí tuyến (Thực tế) : {format_currency(route_metrics['final_total_cost'])}")

print(f"\nThống kê Tổng Chi Phí (Thực tế):")
print(f"  1. Tổng chi phí nhiên liệu: {format_currency(total_final_fuel_cost)}")
print(f"  2. Tổng chi phí lương tài xế: {format_currency(total_final_driver_salary)}")

print("\nPhân tích các điểm chưa được phục vụ:")
if num_unvisited > 0:
    print(f"  Số lượng điểm KH chưa phục vụ: {num_unvisited}")
    unvisited_str = str(unvisited_nodes_list)
    if len(unvisited_str) > 80 : unvisited_str = unvisited_str[:38] + "..." + unvisited_str[-38:]
    print(f"  Danh sách ID: {unvisited_str}")
    print(f"  3. Tổng tiền phạt: {format_currency(total_unvisited_penalty)}")
else:
    print(f"  Tất cả {num_nodes-1} điểm khách hàng đã được phục vụ thành công!")
    print(f"  3. Tổng tiền phạt: {format_currency(0)}")
print(f"\n=============================================")
print(f">>> TỔNG CHI PHÍ MỤC TIÊU: {format_currency(total_final_monetary_cost)} <<<")
print(f"=============================================")
if num_unvisited == 0:
    print("\nTrạng thái giải pháp: Khả thi - Tất cả khách hàng đã được phục vụ 🥰")
else:
    print(f"\nTrạng thái giải pháp: Không khả thi - Còn {num_unvisited} khách hàng chưa được phục vụ 😢")

# =========================================================
# 12. Vẽ đồ thị kết quả
# =========================================================

def plot_routes_hfvrp(coords_plot, routes_info_list_plot, vehicle_types_dict_plot, total_monetary_cost_val, num_unvisited_val, visited_mask, actual_used_counts_plot):
    num_nodes_plot = len(coords_plot)
    plt.figure(figsize=(17, 15)) # Tăng kích thước một chút nữa

    served_node_indices_list = [i for i in range(1, num_nodes_plot) if visited_mask[i]]
    unserved_node_indices = [i for i in range(1, num_nodes_plot) if not visited_mask[i]]

    # --- Vẽ các điểm ---
    if unserved_node_indices:
        unserved_coords = coords_plot[unserved_node_indices]
        plt.scatter(unserved_coords[:, 0], unserved_coords[:, 1], c='silver', marker='x', s=60, label=f'Chưa phục vụ ({len(unserved_node_indices)})', alpha=0.8, zorder=2)
    if served_node_indices_list:
        served_coords = coords_plot[served_node_indices_list]
        plt.scatter(served_coords[:, 0], served_coords[:, 1], c='forestgreen', marker='o', s=50, label=f'Đã phục vụ ({len(served_node_indices_list)})', alpha=0.9, zorder=3)
    plt.scatter(coords_plot[0, 0], coords_plot[0, 1], c='red', marker='s', s=170, label='Kho (Depot)', edgecolors='black', zorder=5) # Đảm bảo kho nổi bật

    # Add node IDs (chỉ khi số nút không quá lớn)
    if num_nodes_plot <= 75: # Tăng ngưỡng nếu muốn hiện ID cho nhiều nút hơn
        for i in range(num_nodes_plot):
            # Slightly offset text to avoid overlapping markers
            plt.text(coords_plot[i, 0] + 2.0, coords_plot[i, 1] + 2.0, str(i), fontsize=6.5, color='dimgray', zorder=4)

    # --- Màu sắc & Kiểu đường cho các tuyến ---
    num_actual_type1 = sum(1 for ri in routes_info_list_plot if ri['type'] == 1)
    num_actual_type2 = sum(1 for ri in routes_info_list_plot if ri['type'] == 2)

    # Dùng colormaps khác biệt hơn, ví dụ 'viridis' cho type 1, 'plasma' cho type 2
    colors_type1 = plt.get_cmap('viridis', max(2, num_actual_type1 + 1)) # +1 để tránh dùng màu cuối cùng quá nhạt nếu chỉ có 1 xe
    colors_type2 = plt.get_cmap('plasma', max(2, num_actual_type2 + 1))
    linestyles = {'owned': '-', 'hired': ':'}
    linewidths = {'owned': 1.5, 'hired': 1.3} # Làm đường thuê mỏng hơn một chút

    # --- Xây dựng Legend Handles ---
    legend_handles = []
    # Markers
    legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label='Kho (Depot)', markerfacecolor='red', markersize=10, markeredgecolor='black'))
    if served_node_indices_list: legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Đã phục vụ ({len(served_node_indices_list)})', markerfacecolor='forestgreen', markersize=8, alpha=0.9))
    if unserved_node_indices: legend_handles.append(plt.Line2D([0], [0], marker='x', color='w', label=f'Chưa phục vụ ({len(unserved_node_indices)})', markerfacecolor='silver', markersize=8, markeredgecolor='darkgray', alpha=0.8))
    # Linestyles
    legend_handles.append(plt.Line2D([0], [0], linestyle='-', color='black', linewidth=1.5, label='Xe Sở hữu'))
    legend_handles.append(plt.Line2D([0], [0], linestyle=':', color='black', linewidth=1.3, label='Xe Thuê'))
    # Add a spacer in legend
    legend_handles.append(plt.Line2D([0], [0], linestyle=' ', label='--- Tuyến Đường ---'))

    # --- Vẽ các tuyến đường ---
    color_idx_type1 = 0
    color_idx_type2 = 0
    legend_line_count = 0
    MAX_LEGEND_ROUTES = 12 # Giảm số tuyến trong legend cho đỡ rối

    for i, info in enumerate(routes_info_list_plot):
        route = info['route']
        if not route or len(route) < 2: continue # Bỏ qua tuyến rỗng hoặc chỉ có kho

        veh_type_id = info['type']
        status = info['status']
        route_coords = coords_plot[route] # Lấy tọa độ các điểm trong tuyến

        label = f'Xe {i+1} (T{veh_type_id},{status[0].upper()})' # VD: T1,O hoặc T2,H
        linestyle = linestyles[status]
        linewidth = linewidths[status]

        # Lấy màu
        if veh_type_id == 1:
            color = colors_type1(color_idx_type1 / max(1, num_actual_type1) if num_actual_type1 > 0 else 0) # Chia cho số lượng thực tế + eps
            color_idx_type1 += 1
        else: # Type 2
            color = colors_type2(color_idx_type2 / max(1, num_actual_type2) if num_actual_type2 > 0 else 0)
            color_idx_type2 += 1


        line, = plt.plot(route_coords[:, 0], route_coords[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, marker='.', markersize=4, label=label, zorder=1, alpha=0.8) # zorder=1 để nằm dưới điểm

        if legend_line_count < MAX_LEGEND_ROUTES:
            legend_handles.append(line)
            legend_line_count += 1
        elif legend_line_count == MAX_LEGEND_ROUTES:
             if len(routes_info_list_plot) > MAX_LEGEND_ROUTES: # Chỉ thêm dòng ... nếu thực sự còn nhiều tuyến
                 legend_handles.append(plt.Line2D([0], [0], linestyle=' ', color='w', label=f'... ({len(routes_info_list_plot)-MAX_LEGEND_ROUTES} tuyến khác)'))
             legend_line_count += 1 # Tăng để không thêm dòng ... nhiều lần

    plt.xlabel("Tọa độ X [km]")
    plt.ylabel("Tọa độ Y [km]")
    used_t1_o_plot = actual_used_counts_plot.get('type1_owned', 0)
    used_t1_h_plot = actual_used_counts_plot.get('type1_hired', 0)
    used_t2_o_plot = actual_used_counts_plot.get('type2_owned', 0)
    used_t2_h_plot = actual_used_counts_plot.get('type2_hired', 0)

    title_str = (f"Kết quả HFVRP - Attr Động + Feature Mới ({num_nodes_plot-1} KH)\n"
                 f"Xe Loại 1 (O/H): {used_t1_o_plot}/{used_t1_h_plot} | Xe Loại 2 (O/H): {used_t2_o_plot}/{used_t2_h_plot}\n"
                 f"Tổng Chi Phí (Thực tế): {format_currency(total_monetary_cost_val)} | Chưa phục vụ: {num_unvisited_val}")
    
    plt.title(title_str, fontsize=12, linespacing=1.5)
    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=8.5, title="Chú giải", title_fontsize='9')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Giữ tỉ lệ trục
    plt.tight_layout(rect=[0, 0, 0.82, 0.95]) # Điều chỉnh rect để có chỗ cho legend và title
    plt.show()

# --- Chạy phần vẽ đồ thị với dữ liệu cuối cùng ---
plot_routes_hfvrp(coords, routes_info, vehicle_types, total_final_monetary_cost, num_unvisited, visited, actual_used_counts)