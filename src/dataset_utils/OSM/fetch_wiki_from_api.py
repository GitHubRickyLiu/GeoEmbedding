"""
从OSM Overpass API获取包含wiki信息的实体
这个脚本可以直接从OSM数据库查询包含wikidata的实体
"""

import requests
import json
import time
from typing import List, Dict
import pandas as pd


class OSMOverpassWikiFetcher:
    """从OSM Overpass API获取包含wiki信息的实体"""
    
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
    def fetch_entities_with_wiki(self, area_name: str = "江西省", 
                                 entity_types: List[str] = None,
                                 timeout: int = 300):
        """
        从Overpass API获取包含wikidata的实体
        
        Args:
            area_name: 区域名称（如"江西省"）
            entity_types: 实体类型列表（如['node', 'way', 'relation']）
            timeout: 超时时间（秒）
            
        Returns:
            Dict: OSM数据
        """
        if entity_types is None:
            entity_types = ['node', 'way', 'relation']
        
        # 构建Overpass QL查询
        query = f"""
        [out:json][timeout:{timeout}];
        area[name="{area_name}"]->.searchArea;
        (
        """
        
        for entity_type in entity_types:
            query += f'  {entity_type}["wikidata"](area.searchArea);\n'
            query += f'  {entity_type}["wikipedia"](area.searchArea);\n'
        
        query += """
        );
        out center;
        """
        
        print(f"正在查询 {area_name} 的OSM数据（包含wiki信息）...")
        print("这可能需要几分钟时间，请耐心等待...\n")
        
        try:
            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"✓ 成功获取 {len(data.get('elements', []))} 条记录")
            return data
            
        except requests.Timeout:
            print("❌ 请求超时，请稍后重试")
            return None
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return None
    
    def parse_osm_data(self, osm_data: Dict) -> List[Dict]:
        """
        解析OSM数据，提取实体信息
        
        Args:
            osm_data: 从Overpass API获取的原始数据
            
        Returns:
            List[Dict]: 解析后的实体列表
        """
        if not osm_data or 'elements' not in osm_data:
            return []
        
        entities = []
        
        for element in osm_data['elements']:
            # 提取坐标
            if element.get('type') == 'node':
                lon = element.get('lon')
                lat = element.get('lat')
            elif 'center' in element:
                lon = element['center'].get('lon')
                lat = element['center'].get('lat')
            else:
                continue
            
            tags = element.get('tags', {})
            
            # 只保留有wiki信息的实体
            wikidata = tags.get('wikidata', '')
            wikipedia = tags.get('wikipedia', '')
            
            if not wikidata and not wikipedia:
                continue
            
            entity = {
                'osm_id': element.get('id'),
                'osm_type': element.get('type'),
                'name': tags.get('name', tags.get('name:zh', tags.get('name:en', 'unnamed'))),
                'longitude': lon,
                'latitude': lat,
                'wikidata': wikidata,
                'wikipedia': wikipedia,
                'wiki_link': wikidata if wikidata else wikipedia,
                'category': tags.get('amenity', tags.get('tourism', tags.get('place', 'unknown')))
            }
            
            entities.append(entity)
        
        print(f"解析得到 {len(entities)} 个包含wiki信息的实体")
        return entities
    
    def save_entities(self, entities: List[Dict], output_prefix: str):
        """
        保存实体数据
        
        Args:
            entities: 实体列表
            output_prefix: 输出文件前缀
        """
        if not entities:
            print("没有数据可保存")
            return
        
        # 保存为JSON
        json_path = f"{output_prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON数据已保存: {json_path}")
        
        # 保存为CSV
        csv_path = f"{output_prefix}.csv"
        df = pd.DataFrame(entities)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV数据已保存: {csv_path}")


def main():
    """主函数"""
    fetcher = OSMOverpassWikiFetcher()
    
    # 方案1: 查询江西省的所有带wiki信息的地点
    print("=" * 60)
    print("从OSM Overpass API获取江西省包含Wiki链接的实体")
    print("=" * 60)
    print()
    
    # 获取数据
    osm_data = fetcher.fetch_entities_with_wiki(
        area_name="江西省",
        entity_types=['node', 'way', 'relation'],
        timeout=300
    )
    
    if osm_data:
        # 解析数据
        entities = fetcher.parse_osm_data(osm_data)
        
        if entities:
            # 保存数据
            output_prefix = r"D:\论文实验\数据集\OSM\jiangxi_wiki_entities"
            fetcher.save_entities(entities, output_prefix)
            
            # 显示统计信息
            print("\n" + "=" * 60)
            print("数据统计")
            print("=" * 60)
            print(f"总实体数: {len(entities)}")
            
            # 按类别统计
            df = pd.DataFrame(entities)
            print("\n类别分布:")
            print(df['category'].value_counts().head(10))
            
            # 显示示例
            print("\n前3条示例数据:")
            for i, entity in enumerate(entities[:3], 1):
                print(f"\n{i}. {entity['name']}")
                print(f"   ID: {entity['osm_id']} ({entity['osm_type']})")
                print(f"   位置: ({entity['longitude']:.6f}, {entity['latitude']:.6f})")
                print(f"   Wiki: {entity['wiki_link']}")
                print(f"   类别: {entity['category']}")


if __name__ == "__main__":
    main()


