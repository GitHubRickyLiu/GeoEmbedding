"""
OSM Shapefile Data Extraction Module
从OSM shapefile文件中提取实体及其相邻实体的信息
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class OSMShapefileExtractor:
    """OSM Shapefile数据提取器"""
    
    def __init__(self, shapefile_path: str, layer: str = None):
        """
        初始化提取器
        
        Args:
            shapefile_path: shapefile文件的路径
            layer: 图层名称，如果为None则使用默认图层
        """
        self.shapefile_path = shapefile_path
        self.layer = layer
        self.gdf = None
        self.filtered_gdf = None
        
    def load_shapefile(self):
        """加载shapefile文件"""
        if self.layer:
            print(f"正在加载shapefile文件: {self.shapefile_path} (图层: {self.layer})")
            try:
                self.gdf = gpd.read_file(self.shapefile_path, layer=self.layer)
                print(f"成功加载 {len(self.gdf)} 条记录")
                print(f"列名: {list(self.gdf.columns)}")
                return True
            except Exception as e:
                print(f"加载shapefile失败: {e}")
                return False
        else:
            print(f"正在加载shapefile文件: {self.shapefile_path}")
            try:
                self.gdf = gpd.read_file(self.shapefile_path)
                print(f"成功加载 {len(self.gdf)} 条记录")
                print(f"列名: {list(self.gdf.columns)}")
                return True
            except Exception as e:
                print(f"加载shapefile失败: {e}")
                return False
    
    def filter_entities_with_wiki(self, use_all_if_no_wiki: bool = False):
        """
        筛选具有wiki链接的实体
        
        Args:
            use_all_if_no_wiki: 如果没有wiki字段，是否使用所有具有名称的实体
        """
        if self.gdf is None:
            print("错误: 请先加载shapefile文件")
            return False
        
        # 查找可能的wiki字段名（可能是wikidata, wikipedia, wiki等）
        wiki_columns = [col for col in self.gdf.columns if 'wiki' in col.lower()]
        
        if not wiki_columns:
            print("警告: 未找到wiki相关字段")
            print(f"可用字段: {list(self.gdf.columns)}")
            
            if use_all_if_no_wiki:
                print("使用所有具有名称的实体...")
                # 筛选有名称的实体
                if 'name' in self.gdf.columns:
                    self.filtered_gdf = self.gdf[self.gdf['name'].notna()].copy()
                    # 添加一个虚拟的wiki字段
                    self.filtered_gdf['wiki_link'] = 'N/A'
                    print(f"筛选后保留 {len(self.filtered_gdf)} 个具有名称的实体")
                    return len(self.filtered_gdf) > 0
            return False
        
        print(f"找到wiki相关字段: {wiki_columns}")
        
        # 筛选：至少有一个wiki字段不为空
        mask = pd.Series([False] * len(self.gdf))
        for col in wiki_columns:
            mask = mask | self.gdf[col].notna()
        
        self.filtered_gdf = self.gdf[mask].copy()
        print(f"筛选后保留 {len(self.filtered_gdf)} 个具有wiki链接的实体")
        
        return len(self.filtered_gdf) > 0
    
    def extract_coordinates(self, geometry):
        """
        从几何对象中提取经纬度
        
        Args:
            geometry: shapely几何对象
            
        Returns:
            tuple: (longitude, latitude)
        """
        if geometry.geom_type == 'Point':
            return geometry.x, geometry.y
        elif geometry.geom_type in ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']:
            # 使用中心点
            centroid = geometry.centroid
            return centroid.x, centroid.y
        else:
            return None, None
    
    def find_nearest_neighbors(self, n_neighbors: int = 5):
        """
        为每个实体找到最近的n个邻居
        
        Args:
            n_neighbors: 需要找到的邻居数量
            
        Returns:
            Dict: 每个实体索引对应的邻居索引列表
        """
        if self.filtered_gdf is None or len(self.filtered_gdf) == 0:
            print("错误: 没有可用的实体数据")
            return None
        
        print(f"正在计算每个实体的最近 {n_neighbors} 个邻居...")
        
        # 提取所有实体的坐标
        coordinates = []
        valid_indices = []
        
        for idx, row in self.filtered_gdf.iterrows():
            lon, lat = self.extract_coordinates(row.geometry)
            if lon is not None and lat is not None:
                coordinates.append([lon, lat])
                valid_indices.append(idx)
        
        if len(coordinates) == 0:
            print("错误: 没有有效的坐标数据")
            return None
        
        coordinates = np.array(coordinates)
        
        # 使用KDTree进行高效的最近邻搜索
        tree = cKDTree(coordinates)
        
        # 查询k+1个最近邻（包括自己），然后排除自己
        k = min(n_neighbors + 1, len(coordinates))
        distances, indices = tree.query(coordinates, k=k)
        
        # 构建邻居映射（排除自己）
        neighbors_map = {}
        for i, idx in enumerate(valid_indices):
            # 排除第一个（自己），取接下来的n_neighbors个
            neighbor_indices = [valid_indices[j] for j in indices[i][1:n_neighbors+1] if j < len(valid_indices)]
            neighbors_map[idx] = neighbor_indices
        
        print(f"成功计算 {len(neighbors_map)} 个实体的邻居关系")
        return neighbors_map
    
    def extract_entity_info(self, row) -> Optional[Dict]:
        """
        提取单个实体的信息
        
        Args:
            row: GeoDataFrame的一行数据
            
        Returns:
            Dict: 实体信息字典
        """
        lon, lat = self.extract_coordinates(row.geometry)
        if lon is None or lat is None:
            return None
        
        # 查找OSM ID字段
        osm_id = None
        for col in ['osm_id', 'id', 'osmid', 'OSM_ID', 'ID']:
            if col in row.index and pd.notna(row[col]):
                osm_id = row[col]
                break
        
        # 查找名称字段
        name = None
        for col in ['name', 'NAME', 'name:zh', 'name:en']:
            if col in row.index and pd.notna(row[col]):
                name = row[col]
                break
        
        # 查找wiki链接字段
        wiki_link = None
        for col in ['wikidata', 'wikipedia', 'wiki', 'WIKIDATA', 'WIKIPEDIA']:
            if col in row.index and pd.notna(row[col]):
                wiki_link = row[col]
                break
        
        return {
            'osm_id': str(osm_id) if osm_id is not None else 'unknown',
            'name': str(name) if name is not None else 'unnamed',
            'longitude': lon,
            'latitude': lat,
            'wiki_link': str(wiki_link) if wiki_link is not None else ''
        }
    
    def extract_all_data(self, n_neighbors: int = 5) -> List[Dict]:
        """
        提取所有实体及其邻居的数据
        
        Args:
            n_neighbors: 每个实体需要提取的邻居数量
            
        Returns:
            List[Dict]: 包含所有实体及其邻居信息的列表
        """
        if self.filtered_gdf is None:
            print("错误: 请先筛选实体数据")
            return []
        
        # 找到最近邻居
        neighbors_map = self.find_nearest_neighbors(n_neighbors)
        if neighbors_map is None:
            return []
        
        print("正在提取实体数据...")
        extracted_data = []
        
        for idx, row in self.filtered_gdf.iterrows():
            # 提取主实体信息
            entity_info = self.extract_entity_info(row)
            if entity_info is None:
                continue
            
            # 获取邻居
            neighbor_indices = neighbors_map.get(idx, [])
            neighbors = []
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in self.filtered_gdf.index:
                    neighbor_row = self.filtered_gdf.loc[neighbor_idx]
                    neighbor_info = self.extract_entity_info(neighbor_row)
                    if neighbor_info is not None:
                        # 邻居不需要wiki链接
                        neighbors.append({
                            'osm_id': neighbor_info['osm_id'],
                            'name': neighbor_info['name'],
                            'longitude': neighbor_info['longitude'],
                            'latitude': neighbor_info['latitude']
                        })
            
            # 组合数据
            data_entry = {
                'entity': entity_info,
                'neighbors': neighbors
            }
            extracted_data.append(data_entry)
        
        print(f"成功提取 {len(extracted_data)} 条数据记录")
        return extracted_data
    
    def save_to_json(self, data: List[Dict], output_path: str):
        """
        将提取的数据保存为JSON文件
        
        Args:
            data: 提取的数据
            output_path: 输出文件路径
        """
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path}")
    
    def save_to_csv(self, data: List[Dict], output_path: str):
        """
        将提取的数据保存为CSV文件（扁平化格式）
        
        Args:
            data: 提取的数据
            output_path: 输出文件路径
        """
        rows = []
        
        for entry in data:
            entity = entry['entity']
            neighbors = entry['neighbors']
            
            row = {
                'osm_id': entity['osm_id'],
                'name': entity['name'],
                'longitude': entity['longitude'],
                'latitude': entity['latitude'],
                'wiki_link': entity['wiki_link']
            }
            
            # 添加邻居信息
            for i, neighbor in enumerate(neighbors[:5], 1):
                row[f'neighbor_{i}_osm_id'] = neighbor['osm_id']
                row[f'neighbor_{i}_name'] = neighbor['name']
                row[f'neighbor_{i}_longitude'] = neighbor['longitude']
                row[f'neighbor_{i}_latitude'] = neighbor['latitude']
            
            # 如果邻居不足5个，填充空值
            for i in range(len(neighbors) + 1, 6):
                row[f'neighbor_{i}_osm_id'] = ''
                row[f'neighbor_{i}_name'] = ''
                row[f'neighbor_{i}_longitude'] = ''
                row[f'neighbor_{i}_latitude'] = ''
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {output_path}")


def main():
    """主函数"""
    # shapefile路径
    shapefile_path = r"D:\论文实验\数据集\OSM\jiangxi-251115-free.shp"
    
    # 使用places图层（包含地点、城市、村庄等信息）
    # 该图层有20,194条记录，其中19,935个有名称（98.7%）
    layer_name = "gis_osm_places_free_1"
    
    print("="*80)
    print("OSM Places数据提取")
    print("="*80)
    print(f"使用图层: {layer_name}")
    print("该图层包含：地点、城市、村庄、城镇等地理位置信息")
    print("="*80)
    print()
    
    # 创建提取器（指定places图层）
    extractor = OSMShapefileExtractor(shapefile_path, layer=layer_name)
    
    # 加载shapefile
    if not extractor.load_shapefile():
        return
    
    # 筛选具有wiki链接的实体（如果没有wiki字段，使用所有有名称的实体）
    if not extractor.filter_entities_with_wiki(use_all_if_no_wiki=True):
        print("\n提示: 该shapefile不包含wiki字段。")
        print("如果需要包含wiki信息的数据，请使用OSM的其他数据源，")
        print("例如从OpenStreetMap API或Overpass API获取数据。")
        return
    
    # 提取所有数据（每个实体及其5个最近邻居）
    print("\n正在提取数据，这可能需要几分钟时间...")
    data = extractor.extract_all_data(n_neighbors=5)
    
    if len(data) == 0:
        print("没有提取到任何数据")
        return
    
    # 保存数据
    output_dir = r"D:\论文实验\数据集\OSM"
    
    # 保存为JSON格式（使用places作为文件名标识）
    json_output = f"{output_dir}\\jiangxi_osm_places_extracted.json"
    extractor.save_to_json(data, json_output)
    
    # 保存为CSV格式
    csv_output = f"{output_dir}\\jiangxi_osm_places_extracted.csv"
    extractor.save_to_csv(data, csv_output)
    
    # 打印数据统计
    print("\n" + "="*80)
    print("数据统计")
    print("="*80)
    print(f"总实体数: {len(data)}")
    
    # 统计有多少实体有完整的5个邻居
    full_neighbors = sum(1 for d in data if len(d['neighbors']) == 5)
    print(f"有完整5个邻居的实体: {full_neighbors} ({full_neighbors/len(data)*100:.1f}%)")
    
    # 打印示例数据
    if len(data) > 0:
        print("\n=== 前3个示例数据 ===")
        for idx in range(min(3, len(data))):
            example = data[idx]
            print(f"\n{idx+1}. 实体: {example['entity']['name']} (ID: {example['entity']['osm_id']})")
            print(f"   位置: ({example['entity']['longitude']:.6f}, {example['entity']['latitude']:.6f})")
            print(f"   邻居数量: {len(example['neighbors'])}")
            for i, neighbor in enumerate(example['neighbors'][:3], 1):
                print(f"     邻居{i}: {neighbor['name']} - ({neighbor['longitude']:.6f}, {neighbor['latitude']:.6f})")
    
    print("\n" + "="*80)
    print("数据提取完成！")
    print("="*80)


if __name__ == "__main__":
    main()

