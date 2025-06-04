import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from datetime import datetime
from scipy.io import savemat
import numpy as np
from sklearn.cluster import DBSCAN
import numpy as np
import scipy.io
import os
import sklearn.linear_model
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.linear_model import RANSACRegressor, LinearRegression
from joblib import Parallel, delayed
def fit_trajectories_theta(data,
                         error_threshold=4,
                         min_points=60,
                         density_radius=1.0,
                         density_threshold=5,
                         max_y_step=20.0,
                         max_time_gap=5.0):
    import numpy as np
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    from sklearn.neighbors import NearestNeighbors

    min_points = max(min_points, 2)
    trajectories_info = []
    data_array = np.asarray(data)
    remaining_data = data_array.copy()
    remaining_indices = np.arange(len(data_array))

    while len(remaining_data) > min_points:
        # Step 1: RANSAC 拟合
        ransac = RANSACRegressor(residual_threshold=error_threshold, min_samples=2)
        X = remaining_data[:, 0].reshape(-1, 1)
        y = remaining_data[:, 1]
        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        inlier_coords = remaining_data[inlier_mask]
        inlier_indices = remaining_indices[inlier_mask]

        if len(inlier_coords) == 0:
            break

        # Step 2: 密度过滤
        nn = NearestNeighbors(radius=density_radius)
        nn.fit(remaining_data)
        densities = nn.radius_neighbors(inlier_coords, return_distance=False)
        density_mask = np.array([len(n) >= density_threshold for n in densities])
        hd_coords = inlier_coords[density_mask]
        hd_indices = inlier_indices[density_mask]

        if len(hd_coords) < min_points:
            remaining_data = remaining_data[~inlier_mask]
            remaining_indices = remaining_indices[~inlier_mask]
            continue

        # Step 3: 时间连续性过滤
        if len(hd_coords) >= 2:
            sort_idx = np.argsort(hd_coords[:, 0])
            sorted_coords = hd_coords[sort_idx]
            sorted_indices = hd_indices[sort_idx]

            time_diffs = np.diff(sorted_coords[:, 0])
            valid_time_mask = np.ones(len(sorted_coords), dtype=bool)
            valid_time_mask[1:] = time_diffs <= max_time_gap

            sorted_coords = sorted_coords[valid_time_mask]
            sorted_indices = sorted_indices[valid_time_mask]

            if len(sorted_coords) < min_points:
                remaining_data = remaining_data[~inlier_mask]
                remaining_indices = remaining_indices[~inlier_mask]
                continue
        else:
            remaining_data = remaining_data[~inlier_mask]
            remaining_indices = remaining_indices[~inlier_mask]
            continue

        # Step 4: 角度跳变 → 分段处理
        y_diffs = np.abs(np.diff(sorted_coords[:, 1]))
        jump_points = np.where(y_diffs > max_y_step)[0] + 1
        segment_bounds = np.concatenate([[0], jump_points, [len(sorted_coords)]])

        # Step 5: 每段轨迹再次拟合并保存
        for i in range(len(segment_bounds) - 1):
            start = segment_bounds[i]
            end = segment_bounds[i + 1]
            segment = sorted_coords[start:end]
            seg_indices = sorted_indices[start:end]

            if len(segment) >= min_points:
                model = LinearRegression()
                model.fit(segment[:, 0].reshape(-1, 1), segment[:, 1])
                slope = model.coef_[0]
                intercept = model.intercept_

                # ===== 新增部分：计算 theta 和 d =====
                # 法向量 (cosθ, sinθ)
                theta = np.arctan(-1.0 / slope) if slope != 0 else np.pi / 2
                # theta = np.arctan(slope)
                # theta = np.degrees(theta)
                # 任意点代入直线：ρ = x*cosθ + y*sinθ
                x0, y0 = segment[0]
                d = x0 * np.cos(theta) + y0 * np.sin(theta)

                trajectories_info.append({
                    "inliers": segment,
                    "slope": slope,
                    "intercept": intercept,
                    "theta": theta,
                    "d": d,
                    "indices": seg_indices.tolist()
                })

                # 移除已拟合点
                mask = np.isin(remaining_indices, seg_indices)
                remaining_data = remaining_data[~mask]
                remaining_indices = remaining_indices[~mask]

    return trajectories_info
def new_fit_trajectories_theta(data,
                                error_threshold=4,
                                min_points=60):  # 参数保留但不使用
    import numpy as np
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    min_points = max(min_points, 2)
    trajectories_info = []
    data_array = np.asarray(data)
    remaining_data = data_array.copy()
    remaining_indices = np.arange(len(data_array))

    while len(remaining_data) > min_points:
        # Step 1: RANSAC 拟合
        ransac = RANSACRegressor(residual_threshold=error_threshold, min_samples=2)
        X = remaining_data[:, 0].reshape(-1, 1)
        y = remaining_data[:, 1]
        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        inlier_coords = remaining_data[inlier_mask]
        inlier_indices = remaining_indices[inlier_mask]

        if len(inlier_coords) < min_points:
            remaining_data = remaining_data[~inlier_mask]
            remaining_indices = remaining_indices[~inlier_mask]
            continue

        # 保留排序（便于可视化和一致性）
        sort_idx = np.argsort(inlier_coords[:, 0])
        segment = inlier_coords[sort_idx]
        seg_indices = inlier_indices[sort_idx]

        # 拟合整段轨迹
        model = LinearRegression()
        model.fit(segment[:, 0].reshape(-1, 1), segment[:, 1])
        slope = model.coef_[0]
        intercept = model.intercept_

        # 计算 theta 和 d
        theta = np.arctan(-1.0 / slope) if slope != 0 else np.pi / 2
        x0, y0 = segment[0]
        d = x0 * np.cos(theta) + y0 * np.sin(theta)

        trajectories_info.append({
            "inliers": segment,
            "slope": slope,
            "intercept": intercept,
            "theta": theta,
            "d": d,
            "indices": seg_indices.tolist()
        })

        # 移除已拟合点
        mask = np.isin(remaining_indices, seg_indices)
        remaining_data = remaining_data[~mask]
        remaining_indices = remaining_indices[~mask]

    return trajectories_info

#读取数据
#input：文件地址
def gen_data(data_name):
    rcParams['font.sans-serif'] = ['SimHei'] 
    rcParams['axes.unicode_minus'] = False  
    data = pd.read_excel(data_name)
    data = data.drop('inde',axis=1)
    data['时间'] = data['时间'].astype(str).str.replace(' ', '')
    data['时间'] = data['时间'].apply(lambda x: x + ":000" if len(x.split(":")) == 3 else x)
    data['时间'] = '2024-11-27 ' + data['时间']
    data['时间'] = pd.to_datetime(data['时间'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    min_time = data['时间'].min()
    data['time'] = (data['时间'] - min_time).dt.total_seconds()  # 转换为秒
    data['data'] = data['真方位']
    data = (
        data
        .sort_values(by='time')          # 按时间升序排序
        .reset_index(drop=True)           # 重置为连续索引
       
    )
    data = data.drop('时间',axis=1)
    data = data.drop('真方位',axis=1)
    
    return data

#拟合ransac直线
#input：data：二维数据 坐标 
# error_threshold：阈值 确定内点 
# min_points：设置直线的最小点数 
# output：rajectories：保存的内点和轨迹
def fit_trajectories(data, error_threshold=4, min_points=60):
    """
    基于 RANSAC 的轨迹分段，输出 theta 和 d。
    theta 为法向角，范围 [0°, 180°)
    """
    trajectories_info = []
    remaining_data = data.copy()
    
    while len(remaining_data) > min_points:
        # 1. RANSAC拟合
        ransac = RANSACRegressor(residual_threshold=error_threshold)
        X = remaining_data[:, 0].reshape(-1, 1)
        y = remaining_data[:, 1]
        ransac.fit(X, y)
        
        # 2. 拿到内点并排序
        inlier_mask = ransac.inlier_mask_
        inliers = remaining_data[inlier_mask]
        sorted_inliers = inliers[np.argsort(inliers[:, 0])]
        
        # 3. 仅保留长度达标的分段
        if len(sorted_inliers) >= min_points:
            model = LinearRegression()
            model.fit(sorted_inliers[:, 0].reshape(-1, 1), sorted_inliers[:, 1])
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # 4. 计算 theta 和 d
            theta_rad = (np.arctan(slope) + np.pi / 2) % np.pi
            theta_deg = np.degrees(theta_rad)
            d = intercept * np.sin(theta_rad)

            # 5. 保存信息
            trajectories_info.append({
                "inliers": sorted_inliers,
                "slope": slope,
                "intercept": intercept,
                "theta": theta_deg,
                "d": d,
                "time_span": (sorted_inliers[0, 0], sorted_inliers[-1, 0])
            })
            
            # 6. 移除已拟合数据
            seg_indices = np.isin(remaining_data[:, 0], sorted_inliers[:, 0]) & \
                          np.isin(remaining_data[:, 1], sorted_inliers[:, 1])
            remaining_data = remaining_data[~seg_indices]
        else:
            break
    
    return trajectories_info

def plot_trajectories(data, trajectories_info):
    """
    绘制 RANSAC 轨迹点和拟合直线。

    参数:
    - data: 原始数据，二维数组，每行包含 [X, Y]。
    - trajectories_info: 每条轨迹的拟合信息列表，包括内点、斜率和截距。
    """
    plt.scatter(data[:, 0], data[:, 1], color='gray', label='Original Data', alpha=0.5)
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, traj in enumerate(trajectories_info):
        # 绘制轨迹点
        inliers = traj["inliers"]
        slope = traj["slope"]
        intercept = traj["intercept"]
        plt.scatter(inliers[:, 0], inliers[:, 1], color=colors[i % len(colors)], label=f'Trajectory {i+1}')

        # 绘制拟合直线
        x_min, x_max = min(inliers[:, 0]), max(inliers[:, 0])
        x_line = np.linspace(x_min, x_max, 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color=colors[i % len(colors)], linestyle='--', label=f'Line {i+1}')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RANSAC Trajectory Segmentation with Fitted Lines')
    plt.show()

#滑窗处理
def convert_one_data_to_region(time_list, angle_list, convert_dt=10 / 1800, t_total=150):

    """
    给 time_list 分区，同时用其结果给 angle_list 分区
    :param time_list: 时间列表
    :param angle_list: 角度列表
    :param convert_dt: 每个区间的时间长度，默认为 10/1800
    :param t_total: 总的时间长度，默认值为 150
    :return: 分区后的时间、角度、坐标列表
    """
    assert len(time_list) == len(angle_list), "时间列表和角度列表长度不一致"

    t_region_start = 0
    end_idx = 0

    convert_time_list = []
    convert_angle_list = []
    convert_coordinate_list = []

    # 按照总时间和区间长度进行划分
    while t_region_start <= t_total:
        temp_time_list, temp_angle_list = [], []
        while end_idx < len(time_list):
            if time_list[end_idx] is None:
                end_idx += 1  # 跳过 None 值
                continue
            if time_list[end_idx] < t_region_start + convert_dt:
                temp_time_list.append(time_list[end_idx])
                temp_angle_list.append(angle_list[end_idx])
                end_idx += 1
            else:
                break

        convert_time_list.append(temp_time_list)
        convert_angle_list.append(temp_angle_list)
        convert_coordinate_list.append([[a, b] for a, b in zip(temp_time_list, temp_angle_list)])

        t_region_start += convert_dt

    return convert_time_list, convert_angle_list, convert_coordinate_list

#聚类
def apply_dbscan_to_regions(convert_time_list, convert_angle_list, eps=0.5, min_samples=2, save_path='data'):

    if not os.path.exists(save_path):
       os.makedirs(save_path)

    region_results = {}  # 保存每个区间的聚类中心值
    for region_idx, (times, angles) in enumerate(zip(convert_time_list, convert_angle_list)):
        
        # 如果区间为空，直接保存一个空文件
        if len(times) == 0:
            print(f"Region {region_idx} is empty, saving empty file.")
            scipy.io.savemat(f"{save_path}/data_{region_idx}.mat", {'cluster_centers': []})
            region_results[region_idx] = []  # 保存空聚类中心值
            continue
        
        # 将时间和角度组合成坐标点，DBSCAN 需要 2D 数据
        points = np.array([[t, a] for t, a in zip(times, angles)])
        
        # 执行 DBSCAN 聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        
        # 获取聚类标签
        labels = db.labels_
        unique_labels = set(labels)
        
        print(f"Region {region_idx}: labels = {labels}")
        print(f"Region {region_idx}: unique_labels = {unique_labels}")
        
        # 计算每个聚类的中心角度值
        cluster_centers = []
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            # 找到属于该簇的点
            cluster_points = points[labels == label]
            # 计算角度的中心值（均值）
            center_angle = cluster_points[:, 1].mean()
            cluster_centers.append(center_angle)
        
        # 如果没有有效的聚类中心，则跳过该区间
        if not cluster_centers:
            print(f"No valid clusters for region {region_idx}, saving empty file.")
            cluster_centers = []  # 保存空聚类中心值
        
        # 保存区间号和聚类中心值到字典
        region_results[region_idx] = cluster_centers
        
        # 将聚类中心值保存为 .mat 文件
        mat_filename = f"{save_path}/data_{region_idx}.mat"
        scipy.io.savemat(mat_filename, {'cluster_centers': cluster_centers})
    
    return region_results

#绘制聚类后的结果
def plot_cluster_centers(dbscan_results):
    """
    绘制聚类中心角度值随区间号的变化图。
    
    :param dbscan_results: DBSCAN 聚类结果，格式为 {区间号: [聚类中心角度值列表]}
    """
    x_vals = []  # 横轴：区间号
    y_vals = []  # 纵轴：聚类中心角度值
    
    for region, centers in dbscan_results.items():
        for center in centers:
            x_vals.append(region)  # 区间号
            y_vals.append(center)  # 角度中心值
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, c='blue', s=1, alpha=0.7, label='Cluster Center')
    plt.title("Cluster Centers Across Regions", fontsize=14)
    plt.xlabel("Region Number", fontsize=12)
    plt.ylabel("Cluster Center Angle Value", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
class ClusterCenter:
    """单个聚类中心及其数据点索引"""
    def __init__(self, center_value, point_indices):
        self.center = center_value
        self.indices = point_indices  # 原数据集的全局索引列表

class ClusterManager:
    """管理某一区域内的所有聚类中心"""
    def __init__(self, region_id, eps=0.5, min_samples=2):
        self.region_id = region_id       # 区域标识符
        self.eps = eps                   # DBSCAN参数
        self.min_samples = min_samples   # DBSCAN参数
        self.clusters = []               # 存储ClusterCenter对象
    
    def add_cluster(self, center, indices):
        """添加一个聚类中心"""
        self.clusters.append(ClusterCenter(center, indices))
    
    def get_clusters(self, min_center=None, max_center=None):
        """获取满足条件的聚类中心"""
        if min_center is None and max_center is None:
            return self.clusters
        return [c for c in self.clusters 
                if (min_center is None or c.center >= min_center) 
                and (max_center is None or c.center <= max_center)]
    
    def save(self, save_dir):
        """保存到.mat文件"""
        os.makedirs(save_dir, exist_ok=True)
        mat_data = {
            'centers': [c.center for c in self.clusters],
            'indices': [c.indices for c in self.clusters],
            'region_id': self.region_id,
            'eps': self.eps,
            'min_samples': self.min_samples
        }
        scipy.io.savemat(
            f"{save_dir}/region_{self.region_id}.mat", 
            mat_data
        )
    
    @classmethod
    def load(cls, file_path):
        """从.mat文件加载"""
        mat_data = scipy.io.loadmat(file_path, simplify_cells=True)
        manager = cls(
            region_id=mat_data['region_id'],
            eps=mat_data.get('eps', 0.5),
            min_samples=mat_data.get('min_samples', 2)
        )
        for center, indices in zip(mat_data['centers'], mat_data['indices']):
            manager.add_cluster(center, indices)
        return manager
def apply_dbscan_to_regions(convert_time_list, convert_angle_list, eps=0.5, 
                           min_samples=2, save_path='data'):
    """
    执行分区域聚类，返回ClusterManager对象字典
    {region_id: ClusterManager实例}
    """
    managers = {}
    for region_idx, (times, angles) in enumerate(zip(convert_time_list, convert_angle_list)):
        # 创建管理器实例
        manager = ClusterManager(
            region_id=region_idx,
            eps=eps,
            min_samples=min_samples
        )
        
        # 处理空数据
        if len(times) == 0:
            manager.save(save_path)
            managers[region_idx] = manager
            continue
        
        # 组合带全局索引的数据点
        global_indices = np.arange(len(times))  # 假设输入数据已全局排序
        points = np.column_stack((times, angles))
        
        # DBSCAN聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        
        # 按簇处理
        for label in set(db.labels_):
            if label == -1: continue  # 跳过噪声
            
            # 提取簇内数据
            mask = (db.labels_ == label)
            cluster_points = points[mask]
            cluster_indices = global_indices[mask].tolist()
            
            # 计算中心值并添加到管理器
            center = cluster_points[:, 1].mean()  # 假设角度在第二列
            manager.add_cluster(center, cluster_indices)
        
        # 保存并记录管理器
        manager.save(save_path)
        managers[region_idx] = manager
    
    return managers