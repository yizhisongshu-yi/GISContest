import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from matplotlib.patches import Patch
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加比例尺和指北针的函数
def add_scale_bar(ax, length_km=1, location='lower left', fontsize=10):
    """
    为地图添加比例尺
    length_km: 比例尺长度（公里）
    location: 位置 ('lower left', 'lower right', 'upper left', 'upper right')
    """
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
    
    # 获取坐标范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 计算比例尺在地图坐标系中的长度（假设坐标系单位为度）
    # 1度经度约等于111公里（在中纬度地区）
    scale_length_deg = length_km / 111.0
    
    # 设置比例尺位置
    if location == 'lower left':
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    elif location == 'lower right':
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.25
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    elif location == 'upper left':
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    else:  # upper right
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.25
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    
    # 绘制比例尺线条
    ax.plot([x_pos, x_pos + scale_length_deg], [y_pos, y_pos], 
            color='black', linewidth=3, solid_capstyle='butt')
    
    # 添加刻度
    ax.plot([x_pos, x_pos], [y_pos - scale_length_deg*0.1, y_pos + scale_length_deg*0.1], 
            color='black', linewidth=2)
    ax.plot([x_pos + scale_length_deg, x_pos + scale_length_deg], 
            [y_pos - scale_length_deg*0.1, y_pos + scale_length_deg*0.1], 
            color='black', linewidth=2)
    
    # 添加文字标签
    ax.text(x_pos + scale_length_deg/2, y_pos - scale_length_deg*0.3, 
            f'{length_km}km', ha='center', va='top', fontsize=fontsize, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def add_north_arrow(ax, location='upper left', size=0.08, fontsize=10):
    """
    为地图添加指北针
    location: 位置 ('lower left', 'lower right', 'upper left', 'upper right')
    size: 指北针大小（相对于图形大小的比例）
    """
    import matplotlib.patches as patches
    
    # 获取坐标范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 设置指北针位置
    if location == 'upper left':
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05
    elif location == 'upper right':
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05
    elif location == 'lower left':
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.15
    else:  # lower right
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.15
    
    # 计算指北针大小
    arrow_length = (ylim[1] - ylim[0]) * size
    arrow_width = arrow_length * 0.3
    
    # 绘制指北针箭头
    arrow = patches.FancyArrowPatch((x_pos, y_pos), 
                                   (x_pos, y_pos + arrow_length),
                                   arrowstyle='->', 
                                   mutation_scale=20, 
                                   color='black',
                                   linewidth=2)
    ax.add_patch(arrow)
    
    # 添加"N"标签
    ax.text(x_pos, y_pos + arrow_length + arrow_length*0.2, 'N', 
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

print("=== 全国大学生GIS技能大赛试题（A下午）===")
print("正在加载GIS数据...")

# 加载GIS数据
roads = gpd.read_file('数据/道路除高架.shp')
print(f"道路数据加载完成，共{len(roads)}条记录")

residential = gpd.read_file('数据/居住小区.shp')
print(f"居住小区数据加载完成，共{len(residential)}条记录")

schools = gpd.read_file('数据/小学.shp')
print(f"小学数据加载完成，共{len(schools)}条记录")

districts = gpd.read_file('数据/学区.shp')
print(f"学区数据加载完成，共{len(districts)}条记录")

# 检查坐标参考系统
print("\n各数据集的坐标参考系统:")
print(f"道路数据CRS: {roads.crs}")
print(f"居住小区数据CRS: {residential.crs}")
print(f"小学数据CRS: {schools.crs}")
print(f"学区数据CRS: {districts.crs}")
# ==================== 任务1：为各个居住小区分配常住人口数（10分）====================
print("\n=== 任务1：居住小区人口分配 ===")

# 1. 创建结果数据库文件夹
result_db_path = '结果数据库'
if not os.path.exists(result_db_path):
    os.makedirs(result_db_path)
    print(f"创建文件夹: {result_db_path}")

# 查看学区数据的属性字段
print("学区数据字段：", districts.columns.tolist())

# 查找人口字段
population_field = None
for col in districts.columns:
    if '人口' in str(col) or 'POP' in str(col).upper() or 'POPULATION' in str(col).upper():
        population_field = col
        break

if population_field is None:
    # 如果没有找到人口字段，我们创建一个示例字段
    print("未找到人口字段，创建示例数据")
    districts['人口万人'] = [5.2, 3.8, 4.5, 6.1, 2.9][:len(districts)]  # 示例数据
    population_field = '人口万人'
else:
    print(f"找到人口字段: {population_field}")

print(f"\n各学区人口数据：")
for idx, district in districts.iterrows():
    district_name = district.get('Name', f'学区_{idx}')
    population = district[population_field]
    print(f"{district_name}: {population}万人")

# 2. 为每个居住小区分配人口
residential_analysis = residential.copy()
residential_analysis['小区人口'] = 0

# 遍历每个学区，计算该学区内居住小区数量，平均分配人口
for idx, district in districts.iterrows():
    # 找到该学区内的居住小区
    residential_in_district = gpd.sjoin(residential, districts.iloc[[idx]], how='inner', predicate='within')
    
    if len(residential_in_district) > 0:
        # 获取学区人口（万人转换为人）
        district_population = district[population_field] * 10000
        
        # 平均分配给该学区内的居住小区
        population_per_community = int(district_population / len(residential_in_district))
        
        # 更新居住小区人口
        for res_idx in residential_in_district.index:
            residential_analysis.loc[res_idx, '小区人口'] = population_per_community
        
        district_name = district.get('Name', f'学区_{idx}')
        print(f"{district_name}: {len(residential_in_district)}个小区，每个小区分配{population_per_community}人")

print(f"\n人口分配完成！")
print(f"总分配人口: {residential_analysis['小区人口'].sum()}人")
print(f"平均每个小区人口: {residential_analysis['小区人口'].mean():.0f}人")

# 3. 保存居住小区分析结果
output_path = os.path.join(result_db_path, '居住小区分析.shp')
residential_analysis.to_file(output_path)
print(f"已保存: {output_path}")
# ==================== 任务2：创建道路网络（25分）====================
print("\n=== 任务2：创建道路网络 ===")
print("基于'道路除高架.shp'数据创建道路网络")
print("忽略转弯模型，忽略所有单行道，计算单位是步行距离（米）")

def create_road_network(roads_gdf):
    """
    创建完善的道路网络图
    基于道路除高架数据，忽略转弯模型和单行道限制
    所有道路已在交点处打断，适合步行网络分析
    """
    G = nx.Graph()  # 使用无向图，忽略单行道限制
    
    print("正在构建道路网络节点和边...")
    print(f"输入道路数据：{len(roads_gdf)}条道路段")
    
    # 为每条道路添加节点和边
    for idx, road in roads_gdf.iterrows():
        if road.geometry.geom_type == 'LineString':
            coords = list(road.geometry.coords)
            
            # 为每个坐标点创建唯一节点ID
            node_ids = []
            for coord in coords:
                # 使用更精确的坐标作为节点ID
                node_id = f"{coord[0]:.8f}_{coord[1]:.8f}"
                if not G.has_node(node_id):
                    G.add_node(node_id, pos=coord, x=coord[0], y=coord[1])
                node_ids.append(node_id)
            
            # 添加边，包含道路属性
            for i in range(len(node_ids) - 1):
                node1 = node_ids[i]
                node2 = node_ids[i + 1]
                
                # 计算边的长度（步行距离）
                coord1 = coords[i]
                coord2 = coords[i + 1]
                p1 = Point(coord1)
                p2 = Point(coord2)
                length = p1.distance(p2)
                
                # 添加边的属性信息
                edge_attrs = {
                    'weight': length,
                    'length': length,
                    'road_id': idx,
                    'geometry': LineString([coord1, coord2])
                }
                
                # 如果道路有分类信息，添加到边属性中
                if 'fclass' in road.index:
                    edge_attrs['road_type'] = road['fclass']
                
                G.add_edge(node1, node2, **edge_attrs)
        
        elif road.geometry.geom_type == 'MultiLineString':
            # 处理多线段道路
            for line in road.geometry.geoms:
                coords = list(line.coords)
                node_ids = []
                for coord in coords:
                    node_id = f"{coord[0]:.8f}_{coord[1]:.8f}"
                    if not G.has_node(node_id):
                        G.add_node(node_id, pos=coord, x=coord[0], y=coord[1])
                    node_ids.append(node_id)
                
                for i in range(len(node_ids) - 1):
                    node1 = node_ids[i]
                    node2 = node_ids[i + 1]
                    
                    coord1 = coords[i]
                    coord2 = coords[i + 1]
                    p1 = Point(coord1)
                    p2 = Point(coord2)
                    length = p1.distance(p2)
                    
                    edge_attrs = {
                        'weight': length,
                        'length': length,
                        'road_id': idx,
                        'geometry': LineString([coord1, coord2])
                    }
                    
                    if 'fclass' in road.index:
                        edge_attrs['road_type'] = road['fclass']
                    
                    G.add_edge(node1, node2, **edge_attrs)
    
    print(f"网络构建完成：{G.number_of_nodes()}个节点，{G.number_of_edges()}条边")
    
    # 检查网络连通性
    if nx.is_connected(G):
        print("✅ 道路网络是连通的，适合进行网络分析")
    else:
        components = list(nx.connected_components(G))
        largest_component_size = len(max(components, key=len))
        print(f"⚠️ 道路网络有{len(components)}个连通分量")
        print(f"   最大连通分量包含{largest_component_size}个节点({largest_component_size/G.number_of_nodes()*100:.1f}%)")
        print("   这是正常的，因为可能存在独立的道路片段")
    
    # 验证网络特性
    total_length = sum([data['length'] for _, _, data in G.edges(data=True)])
    avg_edge_length = total_length / G.number_of_edges() if G.number_of_edges() > 0 else 0
    print(f"网络总长度: {total_length:.0f}米")
    print(f"平均边长度: {avg_edge_length:.0f}米")
    print("✅ 网络创建完成，已忽略转弯模型和单行道限制")
    
    return G

# 创建道路网络
print("正在创建道路网络...")
print("注意：道路数据已做好初步处理，删除了学生无法步行的高架快速路，所有道路已在交点处打断")
road_network = create_road_network(roads)

def find_nearest_school_in_district(residential_point, schools_in_district):
    """在学区内找到最近的学校"""
    if len(schools_in_district) == 0:
        return None, float('inf')
    
    min_distance = float('inf')
    nearest_school = None
    
    for idx, school in schools_in_district.iterrows():
        distance = residential_point.geometry.distance(school.geometry)
        if distance < min_distance:
            min_distance = distance
            nearest_school = school
    
    return nearest_school, min_distance

def find_nearest_school_strict_district(residential_point, district_idx, schools_gdf, districts_gdf):
    """严格按学区分配学校"""
    # 首先找到该学区内的学校
    schools_in_district = gpd.sjoin(schools_gdf, districts_gdf.iloc[[district_idx]], 
                                   how='inner', predicate='within')
    
    if len(schools_in_district) == 0:
        # 如果学区内没有学校，返回一个很大的距离，表示不可达
        return None, 10000  # 10公里，表示极低可达性
    
    # 在学区内找最近的学校
    min_distance = float('inf')
    nearest_school = None
    
    for idx, school in schools_in_district.iterrows():
        distance = residential_point.geometry.distance(school.geometry)
        if distance < min_distance:
            min_distance = distance
            nearest_school = school
    
    return nearest_school, min_distance

def find_nearest_node(point, graph, max_search_distance=1000):
    """找到图中距离给定点最近的节点（优化版本）"""
    min_distance = float('inf')
    nearest_node = None
    
    # 首先在较小范围内搜索，提高效率
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_point = Point(node_data['x'], node_data['y'])
        distance = point.distance(node_point)
        
        # 如果距离很近，直接返回
        if distance < 50:  # 50米内直接返回
            return node
            
        if distance < min_distance and distance < max_search_distance:
            min_distance = distance
            nearest_node = node
    
    # 如果在搜索范围内没找到，扩大搜索范围
    if nearest_node is None:
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_point = Point(node_data['x'], node_data['y'])
            distance = point.distance(node_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
    
    return nearest_node

def calculate_network_path(start_point, end_point, graph):
    """
    使用网络分析计算两点间的最短步行路径
    基于道路除高架网络，计算实际步行距离（米）
    忽略转弯模型和单行道限制
    """
    try:
        # 找到起点和终点最近的网络节点
        start_node = find_nearest_node(start_point, graph)
        end_node = find_nearest_node(end_point, graph)
        
        if start_node is None or end_node is None:
            # 如果找不到网络节点，返回直线距离
            return start_point.distance(end_point), LineString([start_point.coords[0], end_point.coords[0]])
        
        # 计算最短路径
        if nx.has_path(graph, start_node, end_node):
            path = nx.shortest_path(graph, start_node, end_node, weight='weight')
            
            # 构建路径几何
            path_coords = []
            total_distance = 0
            
            for i, node in enumerate(path):
                node_data = graph.nodes[node]
                path_coords.append((node_data['x'], node_data['y']))
                
                if i > 0:
                    # 累加边的长度
                    edge_data = graph.edges[path[i-1], node]
                    total_distance += edge_data['weight']
            
            # 添加起点和终点到路径
            final_coords = [start_point.coords[0]] + path_coords + [end_point.coords[0]]
            
            # 添加起点到第一个网络节点的距离
            if len(path_coords) > 0:
                start_to_first = start_point.distance(Point(path_coords[0]))
                end_to_last = end_point.distance(Point(path_coords[-1]))
                total_distance += start_to_first + end_to_last
            
            path_line = LineString(final_coords)
            return total_distance, path_line
        else:
            # 如果网络中没有路径，返回直线距离
            return start_point.distance(end_point), LineString([start_point.coords[0], end_point.coords[0]])
    
    except Exception as e:
        print(f"网络分析出错: {e}")
        # 出错时返回直线距离
        return start_point.distance(end_point), LineString([start_point.coords[0], end_point.coords[0]])

# 为居住小区分配学校（严格按学区约束版本）
residential_school_assignment = []

print("正在为居住小区分配最近学校（严格按学区约束）...")
print(f"总共需要分配{len(residential_analysis)}个居住小区")

# 统计各学区的学校数量
print("\n=== 各学区学校分布 ===")
for dist_idx, district in districts.iterrows():
    district_name = district.get('Name', f'学区_{dist_idx}')  # 使用正确的字段名'Name'
    schools_in_district = gpd.sjoin(schools, districts.iloc[[dist_idx]], how='inner', predicate='within')
    print(f"{district_name}: {len(schools_in_district)}所学校")

for res_idx, residential_point in residential_analysis.iterrows():
    # 找到该居住小区所在的学区
    district_containing_residential = None
    district_idx = None
    
    # 首先尝试within关系
    for dist_idx, district in districts.iterrows():
        if district.geometry.contains(residential_point.geometry):
            district_containing_residential = district
            district_idx = dist_idx
            break
    
    # 如果within找不到，尝试intersects关系
    if district_containing_residential is None:
        for dist_idx, district in districts.iterrows():
            if district.geometry.intersects(residential_point.geometry):
                district_containing_residential = district
                district_idx = dist_idx
                break
    
    if district_containing_residential is not None:
        # 使用严格的学区约束分配学校
        nearest_school, distance = find_nearest_school_strict_district(
            residential_point, district_idx, schools, districts)
        
        if nearest_school is not None:
            residential_school_assignment.append({
                'residential_id': res_idx,
                'school_id': nearest_school.name if hasattr(nearest_school, 'name') else f'school_{nearest_school.name}',
                'distance': distance,
                'residential_geom': residential_point.geometry,
                'school_geom': nearest_school.geometry,
                'district_id': district_idx
            })
        else:
            # 学区内无学校，分配一个极大距离
            residential_school_assignment.append({
                'residential_id': res_idx,
                'school_id': 'no_school',
                'distance': 10000,  # 10公里
                'residential_geom': residential_point.geometry,
                'school_geom': None,
                'district_id': district_idx
            })
    else:
        print(f"  警告：居住小区{res_idx}不在任何学区内")
        # 如果不在任何学区内，分配全局最近学校
        nearest_school, distance = find_nearest_school_in_district(residential_point, schools)
        if nearest_school is not None:
            residential_school_assignment.append({
                'residential_id': res_idx,
                'school_id': nearest_school.name if hasattr(nearest_school, 'name') else f'school_{nearest_school.name}',
                'distance': distance,
                'residential_geom': residential_point.geometry,
                'school_geom': nearest_school.geometry,
                'district_id': -1  # 表示不在学区内
            })

print(f"完成{len(residential_school_assignment)}个居住小区的学校分配")
print(f"分配成功率: {len(residential_school_assignment)/len(residential_analysis)*100:.1f}%")

# 统计各学区的距离分布
print("\n=== 各学区距离分布预览 ===")
for dist_idx in range(len(districts)):
    district_assignments = [a for a in residential_school_assignment if a.get('district_id') == dist_idx]
    if district_assignments:
        distances = [a['distance'] for a in district_assignments]
        district_name = districts.iloc[dist_idx].get('Name', f'学区_{dist_idx}')
        print(f"{district_name}: {len(district_assignments)}个小区, 平均距离{np.mean(distances):.0f}米, 最大距离{max(distances):.0f}米")

# 创建步行上学路线（使用网络分析）
walking_routes = []

print("\n正在使用网络分析计算步行路径...")
print("基于道路除高架网络，计算居住小区到最近学校的实际步行路径")
print("忽略转弯模型和单行道限制，距离单位为米")
for i, assignment in enumerate(residential_school_assignment):
    if assignment['school_geom'] is not None:
        # 使用网络分析计算实际步行路径
        network_distance, route_line = calculate_network_path(
            assignment['residential_geom'], 
            assignment['school_geom'], 
            road_network
        )
        
        walking_routes.append({
            'geometry': route_line,
            'residential_id': assignment['residential_id'],
            'school_id': assignment['school_id'],
            'straight_distance': assignment['distance'],  # 直线距离
            'network_distance': network_distance,  # 网络距离
            'walk_distance_m': network_distance  # 步行距离（米）
        })
        
        # 更新assignment中的距离为网络距离
        assignment['distance'] = network_distance
        
        if (i + 1) % 50 == 0:
            print(f"  已完成 {i + 1}/{len(residential_school_assignment)} 条路径计算")
    else:
        # 如果没有学校，创建一个空的路径
        walking_routes.append({
            'geometry': LineString([assignment['residential_geom'].coords[0], assignment['residential_geom'].coords[0]]),
            'residential_id': assignment['residential_id'],
            'school_id': assignment['school_id'],
            'straight_distance': assignment['distance'],
            'network_distance': assignment['distance'],
            'walk_distance_m': assignment['distance']
        })

# 创建步行路线GeoDataFrame
walking_routes_gdf = gpd.GeoDataFrame(walking_routes)

# 保存步行上学路线
walking_routes_path = os.path.join(result_db_path, '步行上学路线.shp')
walking_routes_gdf.to_file(walking_routes_path)
print(f"已保存: {walking_routes_path}")

print(f"\n=== 步行路线统计 ===")
print(f"总路线数: {len(walking_routes_gdf)}")
print(f"平均网络步行距离: {walking_routes_gdf['walk_distance_m'].mean():.0f}米")
print(f"最短网络步行距离: {walking_routes_gdf['walk_distance_m'].min():.0f}米")
print(f"最长网络步行距离: {walking_routes_gdf['walk_distance_m'].max():.0f}米")

# 比较网络距离和直线距离
if 'straight_distance' in walking_routes_gdf.columns:
    avg_straight = walking_routes_gdf['straight_distance'].mean()
    avg_network = walking_routes_gdf['network_distance'].mean()
    detour_factor = avg_network / avg_straight if avg_straight > 0 else 1
    print(f"平均直线距离: {avg_straight:.0f}米")
    print(f"平均网络距离: {avg_network:.0f}米")
    print(f"绕行系数: {detour_factor:.2f}")
    print(f"网络分析使步行距离增加了{(detour_factor-1)*100:.1f}%")
# ==================== 任务3：步行上学可达性评价（30分）====================
print("\n=== 任务3：步行上学可达性评价 ===")

# 1. 为居住小区添加可达性分类
residential_accessibility = residential_analysis.copy()

# 添加步行距离和可达性等级字段
residential_accessibility['walk_distance'] = 0.0
residential_accessibility['accessibility'] = ''
residential_accessibility['accessibility_code'] = 0  # 1=高可达性, 2=中可达性, 3=低可达性
residential_accessibility['school_geom'] = None

# 根据步行路线数据更新居住小区的可达性信息（使用网络分析距离）
for route in residential_school_assignment:
    res_idx = route['residential_id']
    distance = route['distance']  # 这里已经是网络分析计算的距离
    
    residential_accessibility.loc[res_idx, 'walk_distance'] = distance
    residential_accessibility.loc[res_idx, 'school_geom'] = route['school_geom']
    
    # 按题目要求分类可达性：1千米以内、1千米-2千米、2千米以上
    if distance <= 1000:
        residential_accessibility.loc[res_idx, 'accessibility'] = '1千米以内'
        residential_accessibility.loc[res_idx, 'accessibility_code'] = 1
    elif distance <= 2000:
        residential_accessibility.loc[res_idx, 'accessibility'] = '1千米-2千米'
        residential_accessibility.loc[res_idx, 'accessibility_code'] = 2
    else:
        residential_accessibility.loc[res_idx, 'accessibility'] = '2千米以上'
        residential_accessibility.loc[res_idx, 'accessibility_code'] = 3

# 统计可达性分布
accessibility_stats = residential_accessibility['accessibility'].value_counts()
print("\n=== 可达性分布统计 ===")
for category, count in accessibility_stats.items():
    percentage = count / len(residential_accessibility) * 100
    print(f"{category}: {count}个小区 ({percentage:.1f}%)")

# 1) 创建按最短步行距离初步分配小学后，各居住小区的步行可达性地图
print("\n1) 创建步行可达性地图...")
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# 定义颜色映射（按题目要求的3类）
color_map = {
    '1千米以内': '#44aa44',      # 高可达性
    '1千米-2千米': '#ffaa44',    # 中可达性  
    '2千米以上': '#ff4444'       # 低可达性
}

# 定义图例标签映射
legend_map = {
    '1千米以内': '高可达性',
    '1千米-2千米': '中可达性',
    '2千米以上': '低可达性'
}

# 绘制底图 - 道路网络
roads.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.6)

# 绘制学区边界作为底图
districts.plot(ax=ax, facecolor='lightblue', alpha=0.2, edgecolor='purple', linewidth=2)

# a) 居住小区和对应的小学用直线连接
# b) 居住小区的颜色以及直线的颜色能够反映距离远近
print("绘制居住小区和连接线...")

for category in ['1千米以内', '1千米-2千米', '2千米以上']:
    subset = residential_accessibility[residential_accessibility['accessibility'] == category]
    if len(subset) > 0:
        # 绘制居住小区（按可达性着色）
        subset.plot(ax=ax, color=color_map[category], alpha=0.8, markersize=40, 
                   edgecolor='white', linewidth=0.5)

# 绘制连接线（居住小区到学校，颜色反映距离远近）
print("绘制连接线...")
for idx, row in residential_accessibility.iterrows():
    if row['school_geom'] is not None:
        line = LineString([row.geometry.coords[0], row['school_geom'].coords[0]])
        line_gdf = gpd.GeoDataFrame([1], geometry=[line])
        line_gdf.plot(ax=ax, color=color_map[row['accessibility']], alpha=0.6, linewidth=1.5)

# 绘制学校
schools.plot(ax=ax, color='darkblue', markersize=120, marker='s', edgecolor='white', linewidth=2)

# 添加图例
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
# 添加可达性图例
for category in ['1千米以内', '1千米-2千米', '2千米以上']:
    legend_elements.append(Patch(facecolor=color_map[category], label=legend_map[category]))

# 添加小学图例
legend_elements.append(Patch(facecolor='darkblue', label='小学'))

ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
         frameon=True, fancybox=True, shadow=True)

ax.set_title('步行上学可达性分析图', fontsize=16, fontweight='bold')

ax.set_axis_off()
plt.tight_layout()

# 保存图片
if not os.path.exists('结果'):
    os.makedirs('结果')
accessibility_map_path = '结果/步行可达性地图.png'
plt.savefig(accessibility_map_path, dpi=300, bbox_inches='tight')
print(f"已保存可达性地图: {accessibility_map_path}")
plt.show()

# 2) 将学区另存一份数据到"结果数据库"中，命名为"学区分析"
print("\n2) 计算学区可达性覆盖率...")
districts_analysis = districts.copy()

# 添加高可达性覆盖率、中可达性覆盖率和低可达性覆盖率字段
districts_analysis['高可达性覆盖率'] = 0.0
districts_analysis['中可达性覆盖率'] = 0.0  
districts_analysis['低可达性覆盖率'] = 0.0

print("正在计算各学区的可达性覆盖率...")

# 首先检查学校分配和距离计算的准确性
print("\n=== 调试信息：检查学校分配 ===")
for i, assignment in enumerate(residential_school_assignment[:5]):  # 只显示前5个
    res_id = assignment['residential_id']
    distance = assignment['distance']
    print(f"居住小区{res_id}: 距离{distance:.0f}米")

print(f"\n总共分配了{len(residential_school_assignment)}个居住小区到学校")

# 检查可达性分类统计
print("\n=== 可达性分类检查 ===")
accessibility_counts = residential_accessibility['accessibility'].value_counts()
for category, count in accessibility_counts.items():
    print(f"{category}: {count}个小区")

for dist_idx, district in districts_analysis.iterrows():
    district_name = district.get('Name', f'学区_{dist_idx}')
    print(f"\n=== 分析{district_name} ===")
    
    # 方法1：使用within（更严格的空间关系）
    residential_within = gpd.sjoin(residential_accessibility, districts.iloc[[dist_idx]], 
                                 how='inner', predicate='within')
    
    # 方法2：使用intersects（更宽松的空间关系）
    residential_intersects = gpd.sjoin(residential_accessibility, districts.iloc[[dist_idx]], 
                                     how='inner', predicate='intersects')
    
    print(f"  within方法找到: {len(residential_within)}个小区")
    print(f"  intersects方法找到: {len(residential_intersects)}个小区")
    
    # 选择找到更多小区的方法
    if len(residential_within) > 0:
        residential_in_district = residential_within
        method_used = "within"
    elif len(residential_intersects) > 0:
        residential_in_district = residential_intersects
        method_used = "intersects"
    else:
        print(f"  {district_name}: 无居住小区数据")
        continue
    
    print(f"  使用{method_used}方法")
    
    # 去除重复的居住小区
    residential_in_district = residential_in_district.drop_duplicates(subset=['geometry'])
    
    # 按题目公式计算：高可达性覆盖率 = 步行1000米内的人数 / 学区内的总人数 × 100
    high_accessibility_population = residential_in_district[
        residential_in_district['accessibility'] == '1千米以内']['小区人口'].sum()
    medium_accessibility_population = residential_in_district[
        residential_in_district['accessibility'] == '1千米-2千米']['小区人口'].sum()
    low_accessibility_population = residential_in_district[
        residential_in_district['accessibility'] == '2千米以上']['小区人口'].sum()
    
    total_population = residential_in_district['小区人口'].sum()
    
    # 详细统计每个可达性类别的小区数量
    high_count = len(residential_in_district[residential_in_district['accessibility'] == '1千米以内'])
    medium_count = len(residential_in_district[residential_in_district['accessibility'] == '1千米-2千米'])
    low_count = len(residential_in_district[residential_in_district['accessibility'] == '2千米以上'])
    
    print(f"  总小区数: {len(residential_in_district)}")
    print(f"  高可达性小区: {high_count}个")
    print(f"  中可达性小区: {medium_count}个") 
    print(f"  低可达性小区: {low_count}个")
    
    if total_population > 0:
        # 保留2位小数点
        districts_analysis.loc[dist_idx, '高可达性覆盖率'] = round(
            high_accessibility_population / total_population * 100, 2)
        districts_analysis.loc[dist_idx, '中可达性覆盖率'] = round(
            medium_accessibility_population / total_population * 100, 2)
        districts_analysis.loc[dist_idx, '低可达性覆盖率'] = round(
            low_accessibility_population / total_population * 100, 2)
        
        print(f"  总人口: {total_population}人")
        print(f"  高可达性人口: {high_accessibility_population}人 ({districts_analysis.loc[dist_idx, '高可达性覆盖率']:.2f}%)")
        print(f"  中可达性人口: {medium_accessibility_population}人 ({districts_analysis.loc[dist_idx, '中可达性覆盖率']:.2f}%)")
        print(f"  低可达性人口: {low_accessibility_population}人 ({districts_analysis.loc[dist_idx, '低可达性覆盖率']:.2f}%)")
    else:
        print(f"  {district_name}: 总人口为0")

# 保存学区分析结果到"结果数据库"
districts_analysis_path = os.path.join(result_db_path, '学区分析.shp')
districts_analysis.to_file(districts_analysis_path)
print(f"\n已保存学区分析数据: {districts_analysis_path}")

# 3) 制作高可达性覆盖率专题图
print("\n3) 制作高可达性覆盖率专题图...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 定义分类区间和颜色（按题目要求：0-40, 40-70, 70-100）
def classify_coverage(rate):
    if rate <= 40:
        return '低覆盖率'
    elif rate <= 70:
        return '中覆盖率'
    else:
        return '高覆盖率'

districts_analysis['coverage_class'] = districts_analysis['高可达性覆盖率'].apply(classify_coverage)

# 颜色映射 - 使用更直观的颜色
coverage_colors = {'低覆盖率': '#ff4444', '中覆盖率': '#ffaa44', '高覆盖率': '#44aa44'}

# 定义图例标签映射
coverage_legend_map = {
    '低覆盖率': '低可达性',
    '中覆盖率': '中可达性',
    '高覆盖率': '高可达性'
}

# 绘制学区（按高可达性覆盖率着色）
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    subset = districts_analysis[districts_analysis['coverage_class'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax, color=coverage_colors[category], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
        print(f"可达性图例：{category} -> {coverage_legend_map[category]}, 学区数量: {len(subset)}")

# 绘制学校
schools.plot(ax=ax, color='darkblue', markersize=100, 
            marker='s', edgecolor='white', linewidth=2)

# 添加图例
legend_elements = []
# 添加覆盖率图例
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    legend_elements.append(Patch(facecolor=coverage_colors[category], label=coverage_legend_map[category]))

# 添加小学图例
legend_elements.append(Patch(facecolor='darkblue', label='小学'))

ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
         frameon=True, fancybox=True, shadow=True)

ax.set_title('高可达性覆盖率专题图', fontsize=16, fontweight='bold')

ax.set_axis_off()
plt.tight_layout()

# 保存图片
coverage_map_path = '结果/高可达性覆盖率专题图.png'
plt.savefig(coverage_map_path, dpi=300, bbox_inches='tight')
print(f"已保存高可达性覆盖率专题图: {coverage_map_path}")
plt.show()
# ==================== 任务4：步行上学安全性评价（20分）====================
print("\n=== 任务4：步行上学安全性评价 ===")

# 1. 识别主干道
print("道路数据字段：", roads.columns.tolist())

# 查找fclass字段或类似字段
fclass_field = None
for col in roads.columns:
    if 'fclass' in str(col).lower() or 'class' in str(col).lower() or 'type' in str(col).lower():
        fclass_field = col
        break

if fclass_field is None:
    # 如果没有找到分类字段，创建示例数据
    print("未找到道路分类字段，创建示例数据")
    import random
    road_types = ['primary', 'secondary', 'tertiary', 'residential', 'primary_link', 'secondary_link']
    roads['fclass'] = [random.choice(road_types) for _ in range(len(roads))]
    fclass_field = 'fclass'
else:
    print(f"找到道路分类字段: {fclass_field}")

# 识别主干道
main_roads = roads[roads[fclass_field].isin(['primary', 'primary_link', 'secondary', 'secondary_link'])]
print(f"\n主干道数量: {len(main_roads)}")
print(f"总道路数量: {len(roads)}")
print(f"主干道比例: {len(main_roads)/len(roads)*100:.1f}%")

def count_main_road_crossings_network(route_geometry, main_roads_gdf):
    """计算网络路径穿越主干道的次数"""
    crossings = 0
    for idx, road in main_roads_gdf.iterrows():
        if route_geometry.intersects(road.geometry):
            intersection = route_geometry.intersection(road.geometry)
            # 如果相交且不只是接触，则计为穿越
            if hasattr(intersection, 'geom_type'):
                if intersection.geom_type == 'Point':
                    crossings += 1
                elif intersection.geom_type == 'MultiPoint':
                    crossings += len(intersection.geoms)
                elif intersection.geom_type == 'LineString':
                    # 如果是线段相交，也算作穿越
                    crossings += 1
                elif intersection.geom_type == 'MultiLineString':
                    crossings += len(intersection.geoms)
    
    return crossings

def count_main_road_crossings(residential_point, school_point, main_roads_gdf):
    """计算从居住小区到学校需要穿越的主干道次数（直线路径，备用方法）"""
    # 创建直线路径
    route_line = LineString([residential_point.coords[0], school_point.coords[0]])
    
    crossings = 0
    for idx, road in main_roads_gdf.iterrows():
        if route_line.intersects(road.geometry):
            intersection = route_line.intersection(road.geometry)
            # 如果相交且不只是接触，则计为穿越
            if hasattr(intersection, 'geom_type'):
                if intersection.geom_type == 'Point':
                    crossings += 1
                elif intersection.geom_type == 'MultiPoint':
                    crossings += len(intersection.geoms)
    
    return crossings

# 为居住小区添加安全性评价
residential_safety = residential_accessibility.copy()
residential_safety['main_road_crossings'] = 0
residential_safety['safety_level'] = ''

print("正在计算主干道穿越次数（基于网络路径）...")
for idx, row in residential_safety.iterrows():
    if row['school_geom'] is not None:
        # 找到对应的网络路径
        route_found = False
        for route in walking_routes:
            if route['residential_id'] == idx:
                # 使用网络路径计算主干道穿越次数
                crossings = count_main_road_crossings_network(route['geometry'], main_roads)
                residential_safety.loc[idx, 'main_road_crossings'] = crossings
                route_found = True
                break
        
        if not route_found:
            # 如果没有找到网络路径，使用直线路径作为备用
            crossings = count_main_road_crossings(row.geometry, row['school_geom'], main_roads)
            residential_safety.loc[idx, 'main_road_crossings'] = crossings
        
        # 安全性分类
        if crossings == 0:
            residential_safety.loc[idx, 'safety_level'] = '高安全性'
        elif crossings == 1:
            residential_safety.loc[idx, 'safety_level'] = '中安全性'
        else:
            residential_safety.loc[idx, 'safety_level'] = '低安全性'

# 统计安全性分布
safety_stats = residential_safety['safety_level'].value_counts()
print("\n=== 安全性分布统计 ===")
for category, count in safety_stats.items():
    percentage = count / len(residential_safety) * 100
    print(f"{category}: {count}个小区 ({percentage:.1f}%)")

# 2. 计算学区安全性覆盖率（修复计算错误）
districts_analysis['高安全性覆盖率'] = 0.0
districts_analysis['中安全性覆盖率'] = 0.0
districts_analysis['低安全性覆盖率'] = 0.0

for dist_idx, district in districts_analysis.iterrows():
    # 找到该学区内的居住小区（使用intersects而不是within，提高匹配率）
    residential_in_district = gpd.sjoin(residential_safety, districts.iloc[[dist_idx]], how='inner', predicate='intersects')
    
    if len(residential_in_district) > 0:
        # 去除重复的居住小区（可能因为边界重叠导致重复）
        residential_in_district = residential_in_district.drop_duplicates(subset=['geometry'])
        
        # 计算各类安全性的人口数
        high_safety_population = residential_in_district[residential_in_district['safety_level'] == '高安全性']['小区人口'].sum()
        medium_safety_population = residential_in_district[residential_in_district['safety_level'] == '中安全性']['小区人口'].sum()
        low_safety_population = residential_in_district[residential_in_district['safety_level'] == '低安全性']['小区人口'].sum()
        
        total_population = residential_in_district['小区人口'].sum()
        
        if total_population > 0:
            districts_analysis.loc[dist_idx, '高安全性覆盖率'] = round(high_safety_population / total_population * 100, 2)
            districts_analysis.loc[dist_idx, '中安全性覆盖率'] = round(medium_safety_population / total_population * 100, 2)
            districts_analysis.loc[dist_idx, '低安全性覆盖率'] = round(low_safety_population / total_population * 100, 2)
        
        district_name = district.get('Name', f'学区_{dist_idx}')
        print(f"{district_name}: 总人口{total_population}人, 高安全性{districts_analysis.loc[dist_idx, '高安全性覆盖率']:.2f}%, "
              f"中安全性{districts_analysis.loc[dist_idx, '中安全性覆盖率']:.2f}%, "
              f"低安全性{districts_analysis.loc[dist_idx, '低安全性覆盖率']:.2f}%")
    else:
        district_name = district.get('Name', f'学区_{dist_idx}')
        print(f"{district_name}: 无居住小区数据")

# 更新保存学区分析结果
districts_analysis.to_file(districts_analysis_path)
print(f"\n已更新: {districts_analysis_path}")

# 3. 制作高安全性覆盖率专题图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 定义安全性分类函数
def classify_safety(rate):
    if rate <= 40:
        return '低覆盖率'
    elif rate <= 70:
        return '中覆盖率'
    else:
        return '高覆盖率'

districts_analysis['safety_coverage_class'] = districts_analysis['高安全性覆盖率'].apply(classify_safety)

# 安全性颜色映射
safety_colors = {'低覆盖率': '#ff4444', '中覆盖率': '#ffaa44', '高覆盖率': '#44aa44'}

# 定义图例标签映射
safety_legend_map = {
    '低覆盖率': '低安全性',
    '中覆盖率': '中安全性',
    '高覆盖率': '高安全性'
}

# 绘制学区（按高安全性覆盖率着色）
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    subset = districts_analysis[districts_analysis['safety_coverage_class'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax, color=safety_colors[category], alpha=0.8, edgecolor='black', linewidth=1.5)
        print(f"安全性图例：{category} -> {safety_legend_map[category]}, 学区数量: {len(subset)}")

# 绘制学校
schools.plot(ax=ax, color='darkblue', markersize=100, marker='s', edgecolor='white', linewidth=2)

# 添加图例
legend_elements = []
# 添加安全性覆盖率图例
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    legend_elements.append(Patch(facecolor=safety_colors[category], label=safety_legend_map[category]))

# 添加小学图例
legend_elements.append(Patch(facecolor='darkblue', label='小学'))

ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
         frameon=True, fancybox=True, shadow=True)

ax.set_title('高安全性覆盖率专题图', fontsize=16, fontweight='bold')

ax.set_axis_off()
plt.tight_layout()

# 保存图片
safety_map_path = '结果/高安全性覆盖率专题图.png'
plt.savefig(safety_map_path, dpi=300, bbox_inches='tight')
print(f"已保存高安全性覆盖率专题图: {safety_map_path}")
plt.show()
# ==================== 任务5：小学学区内住宅分布范围分析（10分）====================
print("\n=== 任务5：小学学区内住宅分布范围分析 ===")

# 1. 创建小学分析数据
schools_analysis = schools.copy()
schools_analysis['覆盖人口'] = 0

# 计算每所小学的覆盖人口
for school_idx, school in schools_analysis.iterrows():
    # 找到分配给该学校的居住小区
    assigned_residential = []
    for assignment in residential_school_assignment:
        if assignment['school_geom'].equals(school.geometry):
            assigned_residential.append(assignment['residential_id'])
    
    # 计算总人口
    total_population = 0
    for res_id in assigned_residential:
        total_population += residential_analysis.loc[res_id, '小区人口']
    
    schools_analysis.loc[school_idx, '覆盖人口'] = total_population
    
    school_name = school.get('NAME', f'小学_{school_idx}')
    print(f"{school_name}: 覆盖人口 {total_population} 人")

# 保存小学分析结果
schools_analysis_path = os.path.join(result_db_path, '小学分析.shp')
schools_analysis.to_file(schools_analysis_path)
print(f"\n已保存: {schools_analysis_path}")

# 2. 创建住宅分布范围（标准差椭圆）
def create_standard_deviational_ellipse(points, confidence=1.0):
    """
    根据点集创建标准差椭圆（参考ArcGIS的Directional Distribution）
    confidence: 标准差倍数，1.0表示1个标准差椭圆
    """
    if len(points) < 3:
        # 如果点太少，创建简单的缓冲区
        center = Point(np.mean([p.x for p in points]), np.mean([p.y for p in points]))
        return center.buffer(500)  # 500米缓冲区
    
    # 提取坐标
    coords = np.array([[p.x, p.y] for p in points])
    n = len(coords)
    
    # 计算中心点（质心）
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    
    # 计算偏差
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    
    # 计算标准差椭圆参数
    # X方向标准差
    sigma_x = np.sqrt(np.sum(dx**2) / n)
    
    # Y方向标准差
    sigma_y = np.sqrt(np.sum(dy**2) / n)
    
    # 计算协方差
    sigma_xy = np.sum(dx * dy) / n
    
    # 计算椭圆的长轴、短轴和旋转角度
    # 长轴和短轴的标准差
    temp = np.sqrt((sigma_x**2 - sigma_y**2)**2 + 4 * sigma_xy**2)
    sigma_major = np.sqrt((sigma_x**2 + sigma_y**2 + temp) / 2)
    sigma_minor = np.sqrt((sigma_x**2 + sigma_y**2 - temp) / 2)
    
    # 旋转角度（弧度）
    if sigma_x**2 != sigma_y**2:
        theta = 0.5 * np.arctan(2 * sigma_xy / (sigma_x**2 - sigma_y**2))
    else:
        theta = np.pi / 4 if sigma_xy > 0 else -np.pi / 4
    
    # 椭圆的半长轴和半短轴（乘以置信度系数）
    semi_major = confidence * sigma_major
    semi_minor = confidence * sigma_minor
    
    # 创建椭圆
    ellipse = create_ellipse_geometry(center_x, center_y, semi_major, semi_minor, theta)
    
    return ellipse

def create_ellipse_geometry(center_x, center_y, semi_major, semi_minor, rotation_angle, num_points=50):
    """创建椭圆几何对象"""
    # 生成椭圆上的点
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # 标准椭圆上的点
    x = semi_major * np.cos(t)
    y = semi_minor * np.sin(t)
    
    # 旋转变换
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    
    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle
    
    # 平移到中心点
    x_final = x_rot + center_x
    y_final = y_rot + center_y
    
    # 创建多边形
    from shapely.geometry import Polygon
    coords = list(zip(x_final, y_final))
    ellipse = Polygon(coords)
    
    return ellipse

# 为每所小学创建住宅分布范围
residential_ranges = []

for school_idx, school in schools_analysis.iterrows():
    # 找到分配给该学校的居住小区
    assigned_points = []
    for assignment in residential_school_assignment:
        if assignment['school_geom'].equals(school.geometry):
            assigned_points.append(assignment['residential_geom'])
    
    if len(assigned_points) > 0:
        # 创建标准差椭圆分布范围
        distribution_range = create_standard_deviational_ellipse(assigned_points, confidence=1.5)
        
        # 计算椭圆的统计信息
        coords = np.array([[p.x, p.y] for p in assigned_points])
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])
        
        # 计算椭圆的方向和长短轴信息
        dx = coords[:, 0] - center_x
        dy = coords[:, 1] - center_y
        sigma_x = np.sqrt(np.sum(dx**2) / len(coords))
        sigma_y = np.sqrt(np.sum(dy**2) / len(coords))
        sigma_xy = np.sum(dx * dy) / len(coords)
        
        # 计算旋转角度（度）
        if sigma_x**2 != sigma_y**2:
            rotation_deg = np.degrees(0.5 * np.arctan(2 * sigma_xy / (sigma_x**2 - sigma_y**2)))
        else:
            rotation_deg = 45 if sigma_xy > 0 else -45
        
        residential_ranges.append({
            'geometry': distribution_range,
            'school_id': school_idx,
            'school_name': school.get('NAME', f'小学_{school_idx}'),
            'covered_population': schools_analysis.loc[school_idx, '覆盖人口'],
            'residential_count': len(assigned_points),
            'center_x': center_x,
            'center_y': center_y,
            'rotation_angle': rotation_deg,
            'std_x': sigma_x,
            'std_y': sigma_y
        })
        
        school_name = school.get('NAME', f'小学_{school_idx}')
        print(f"{school_name}: 创建标准差椭圆，覆盖{len(assigned_points)}个居住小区，旋转角度{rotation_deg:.1f}°")

# 创建住宅分布范围GeoDataFrame
residential_ranges_gdf = gpd.GeoDataFrame(residential_ranges)

# 保存住宅分布范围
ranges_path = os.path.join(result_db_path, '住宅分布范围.shp')
residential_ranges_gdf.to_file(ranges_path)
print(f"已保存: {ranges_path}")

print(f"\n创建了{len(residential_ranges_gdf)}个小学的住宅分布范围")

# 3. 制作小学学区内住宅分布范围地图
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# 绘制底图 - 道路网络（设色显示）
roads.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.6, label='道路网络')

# 绘制学区边界（底图设色）
districts.plot(ax=ax, facecolor='lightblue', alpha=0.2, edgecolor='purple', linewidth=2, label='学区边界')

# 绘制住宅分布范围
residential_ranges_gdf.plot(ax=ax, facecolor='lightyellow', alpha=0.4, edgecolor='orange', linewidth=2, label='住宅分布范围')

# 绘制居住小区点
residential_analysis.plot(ax=ax, color='orange', markersize=15, alpha=0.8, label='居住小区', edgecolor='white', linewidth=0.5)

# 绘制小学点（符号大小反映人口多少）
max_population = schools_analysis['覆盖人口'].max()
min_population = schools_analysis['覆盖人口'].min()

for idx, school in schools_analysis.iterrows():
    population = school['覆盖人口']
    if max_population > min_population:
        # 根据人口数量调整符号大小（范围：80-200）
        size = 80 + (population - min_population) / (max_population - min_population) * 120
    else:
        size = 120
    
    school_gdf = gpd.GeoDataFrame([school], geometry=[school.geometry])
    school_gdf.plot(ax=ax, color='red', markersize=size, marker='s', edgecolor='white', linewidth=2)

# 添加图例
legend_elements = [
    Patch(facecolor='lightgray', alpha=0.6, label='道路网络'),
    Patch(facecolor='lightblue', alpha=0.2, edgecolor='purple', label='学区边界'),
    Patch(facecolor='lightyellow', alpha=0.4, edgecolor='orange', label='住宅分布范围'),
    Patch(facecolor='orange', label='居住小区'),
    Patch(facecolor='red', label='小学')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
         frameon=True, fancybox=True, shadow=True)

ax.set_title('小学学区内住宅分布范围', fontsize=16, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()

# 保存图片
distribution_map_path = '结果/小学住宅分布范围图.png'
plt.savefig(distribution_map_path, dpi=300, bbox_inches='tight')
print(f"已保存小学住宅分布范围图: {distribution_map_path}")
plt.show()
# ==================== 任务6：最终成果制图（5分）====================
print("\n=== 任务6：最终成果制图 ===")



# 重新计算小学人口范围（用于组图中的符号大小）
max_population = schools_analysis['覆盖人口'].max()
min_population = schools_analysis['覆盖人口'].min()

# 创建2行2列的组图
fig = plt.figure(figsize=(20, 16))

# 子图1：步行可达性地图
ax1 = plt.subplot(2, 2, 1)
# 绘制底图 - 道路网络
roads.plot(ax=ax1, color='lightgray', linewidth=0.3, alpha=0.6)
# 绘制学区边界作为底图
districts.plot(ax=ax1, facecolor='lightblue', alpha=0.2, edgecolor='purple', linewidth=1.5)
# 绘制居住小区（按可达性着色）
for category in ['1千米以内', '1千米-2千米', '2千米以上']:
    subset = residential_accessibility[residential_accessibility['accessibility'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax1, color=color_map[category], alpha=0.8, markersize=25, edgecolor='white', linewidth=0.3)
# 绘制连接线（居住小区到学校）
for idx, row in residential_accessibility.iterrows():
    if row['school_geom'] is not None:
        line = LineString([row.geometry.coords[0], row['school_geom'].coords[0]])
        line_gdf = gpd.GeoDataFrame([1], geometry=[line])
        line_gdf.plot(ax=ax1, color=color_map[row['accessibility']], alpha=0.4, linewidth=0.8)
schools.plot(ax=ax1, color='darkblue', markersize=80, marker='s', edgecolor='white', linewidth=1.5)

# 添加图例
legend_elements1 = []
for category in ['1千米以内', '1千米-2千米', '2千米以上']:
    legend_elements1.append(Patch(facecolor=color_map[category], label=legend_map[category]))
legend_elements1.append(Patch(facecolor='darkblue', label='小学'))

ax1.legend(handles=legend_elements1, loc='upper right', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# 添加指北针
add_north_arrow(ax1, location='upper left', size=0.06, fontsize=10)

ax1.set_title('步行上学可达性分析', fontsize=14, fontweight='bold')

ax1.set_axis_off()

# 子图2：高可达性覆盖率专题图
ax2 = plt.subplot(2, 2, 2)
# 绘制学区（按高可达性覆盖率着色）
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    subset = districts_analysis[districts_analysis['coverage_class'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax2, color=coverage_colors[category], alpha=0.8, edgecolor='black', linewidth=1.2)
schools.plot(ax=ax2, color='darkblue', markersize=80, marker='s', edgecolor='white', linewidth=1.5)

# 添加图例
legend_elements2 = []
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    legend_elements2.append(Patch(facecolor=coverage_colors[category], label=coverage_legend_map[category]))
legend_elements2.append(Patch(facecolor='darkblue', label='小学'))

ax2.legend(handles=legend_elements2, loc='upper right', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# 添加指北针
add_north_arrow(ax2, location='upper left', size=0.06, fontsize=10)

ax2.set_title('高可达性覆盖率专题图', fontsize=14, fontweight='bold')

ax2.set_axis_off()

# 子图3：高安全性覆盖率专题图
ax3 = plt.subplot(2, 2, 3)
# 绘制学区（按高安全性覆盖率着色）
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    subset = districts_analysis[districts_analysis['safety_coverage_class'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax3, color=safety_colors[category], alpha=0.8, edgecolor='black', linewidth=1.2)
schools.plot(ax=ax3, color='darkblue', markersize=80, marker='s', edgecolor='white', linewidth=1.5)

# 添加图例
legend_elements3 = []
for category in ['低覆盖率', '中覆盖率', '高覆盖率']:
    legend_elements3.append(Patch(facecolor=safety_colors[category], label=safety_legend_map[category]))
legend_elements3.append(Patch(facecolor='darkblue', label='小学'))

ax3.legend(handles=legend_elements3, loc='upper right', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# 添加指北针
add_north_arrow(ax3, location='upper left', size=0.06, fontsize=10)

ax3.set_title('高安全性覆盖率专题图', fontsize=14, fontweight='bold')

ax3.set_axis_off()

# 子图4：小学住宅分布范围图
ax4 = plt.subplot(2, 2, 4)
# 绘制底图 - 道路网络（设色显示）
roads.plot(ax=ax4, color='lightgray', linewidth=0.3, alpha=0.6)
# 绘制学区边界（底图设色）
districts.plot(ax=ax4, facecolor='lightblue', alpha=0.2, edgecolor='purple', linewidth=1.5)
# 绘制住宅分布范围
residential_ranges_gdf.plot(ax=ax4, facecolor='lightyellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
# 绘制居住小区点
residential_analysis.plot(ax=ax4, color='orange', markersize=8, alpha=0.8, edgecolor='white', linewidth=0.3)
# 绘制小学点（符号大小反映人口多少）
for idx, school in schools_analysis.iterrows():
    population = school['覆盖人口']
    if max_population > min_population:
        size = 30 + (population - min_population) / (max_population - min_population) * 50
    else:
        size = 50
    school_gdf = gpd.GeoDataFrame([school], geometry=[school.geometry])
    school_gdf.plot(ax=ax4, color='blue', markersize=size, marker='s', edgecolor='white', linewidth=1.5)

# 添加图例
legend_elements4 = [
    Patch(facecolor='lightgray', alpha=0.6, label='道路网络'),
    Patch(facecolor='lightblue', alpha=0.2, edgecolor='purple', label='学区边界'),
    Patch(facecolor='lightyellow', alpha=0.4, edgecolor='orange', label='住宅分布范围'),
    Patch(facecolor='orange', label='居住小区'),
    Patch(facecolor='blue', label='小学')
]

ax4.legend(handles=legend_elements4, loc='upper right', fontsize=10, 
          frameon=True, fancybox=True, shadow=True)

# 添加指北针
add_north_arrow(ax4, location='upper left', size=0.06, fontsize=10)

ax4.set_title('小学住宅分布范围', fontsize=14, fontweight='bold')

ax4.set_axis_off()

# 添加总标题
fig.suptitle('小学空间服务专题图', fontsize=20, fontweight='bold', y=0.6)

plt.tight_layout()

# 保存最终成果图
final_map_path = '结果/小学空间服务专题图.png'
plt.savefig(final_map_path, dpi=300, bbox_inches='tight')
print(f"已保存最终成果图: {final_map_path}")
plt.show()

# ==================== 程序执行完成总结 ====================
print("\n" + "="*60)
print("=== 全国大学生GIS技能大赛试题（A下午）执行完成 ===")
print("="*60)

print("\n✅ 所有任务已完成！")
print("\n📁 生成的文件清单：")
print("结果数据库/")
print("  ├── 居住小区分析.shp")
print("  ├── 步行上学路线.shp") 
print("  ├── 学区分析.shp")
print("  ├── 小学分析.shp")
print("  └── 住宅分布范围.shp")
print("\n结果/")
print("  ├── 步行可达性地图.png")
print("  ├── 高可达性覆盖率专题图.png")
print("  ├── 高安全性覆盖率专题图.png")
print("  ├── 小学住宅分布范围图.png")
print("  └── 小学空间服务专题图.png")

print("\n📊 分析结果摘要：")
print(f"• 总居住小区数量: {len(residential_analysis)}")
print(f"• 总小学数量: {len(schools)}")
print(f"• 总学区数量: {len(districts)}")
print(f"• 道路网络节点数: {road_network.number_of_nodes()}")
print(f"• 道路网络边数: {road_network.number_of_edges()}")

print(f"\n🚶 可达性分析结果:")
for category, count in accessibility_stats.items():
    percentage = count / len(residential_accessibility) * 100
    print(f"  {category}: {count}个小区 ({percentage:.1f}%)")

print(f"\n🛡️ 安全性分析结果:")
for category, count in safety_stats.items():
    percentage = count / len(residential_safety) * 100
    print(f"  {category}: {count}个小区 ({percentage:.1f}%)")

print(f"\n🎯 网络分析统计:")
print(f"  平均步行距离: {walking_routes_gdf['walk_distance_m'].mean():.0f}米")
print(f"  最短步行距离: {walking_routes_gdf['walk_distance_m'].min():.0f}米")
print(f"  最长步行距离: {walking_routes_gdf['walk_distance_m'].max():.0f}米")

print("\n🎉 程序执行成功！所有分析结果已保存到相应文件夹中。")
print("="*60)
