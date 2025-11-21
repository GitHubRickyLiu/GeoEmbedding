"""
快速检查所有图层中的字段，特别是wiki相关字段
"""

import geopandas as gpd
import fiona
import sys
import io

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def quick_check_all_layers(shapefile_path: str):
    """
    快速检查所有图层的字段信息，重点查找wiki字段
    
    Args:
        shapefile_path: shapefile文件路径
    """
    print("=" * 100)
    print("快速检查OSM Shapefile所有图层")
    print("=" * 100)
    print(f"文件: {shapefile_path}\n")
    
    try:
        # 获取所有图层
        layers = fiona.listlayers(shapefile_path)
        print(f"共找到 {len(layers)} 个图层\n")
        
        wiki_keywords = ['wiki', 'wikidata', 'wikipedia', 'url', 'link', 'web', 'website', 'ref', 'source']
        
        # 存储结果
        results = []
        
        for i, layer in enumerate(layers, 1):
            print(f"[{i}/{len(layers)}] 检查图层: {layer}")
            
            try:
                # 读取图层
                gdf = gpd.read_file(shapefile_path, layer=layer)
                record_count = len(gdf)
                
                # 获取所有字段（除了geometry）
                fields = [col for col in gdf.columns if col != 'geometry']
                
                # 查找wiki相关字段
                wiki_fields = []
                for col in fields:
                    for keyword in wiki_keywords:
                        if keyword in col.lower():
                            # 检查非空值数量
                            non_null = gdf[col].notna().sum()
                            if non_null > 0:
                                wiki_fields.append(f"{col}({non_null})")
                            break
                
                # 检查是否有name字段
                name_fields = [col for col in fields if 'name' in col.lower()]
                name_count = 0
                if name_fields:
                    name_count = gdf[name_fields[0]].notna().sum()
                
                result = {
                    'layer': layer,
                    'records': record_count,
                    'fields': len(fields),
                    'field_names': fields,
                    'wiki_fields': wiki_fields,
                    'name_count': name_count
                }
                results.append(result)
                
                # 输出简要信息
                print(f"  记录数: {record_count:,}")
                print(f"  字段数: {len(fields)}")
                print(f"  字段列表: {', '.join(fields)}")
                
                if wiki_fields:
                    print(f"  *** 找到Wiki字段: {', '.join(wiki_fields)} ***")
                else:
                    print(f"  Wiki字段: 无")
                
                if name_count > 0:
                    print(f"  有名称的记录: {name_count:,} ({name_count/record_count*100:.1f}%)")
                
                print()
                
            except Exception as e:
                print(f"  [错误] 无法读取图层: {e}\n")
        
        # 汇总报告
        print("\n" + "=" * 100)
        print("汇总报告")
        print("=" * 100)
        
        print("\n包含Wiki相关字段的图层:")
        wiki_layers = [r for r in results if r['wiki_fields']]
        if wiki_layers:
            for r in wiki_layers:
                print(f"\n图层: {r['layer']}")
                print(f"  记录数: {r['records']:,}")
                print(f"  Wiki字段: {', '.join(r['wiki_fields'])}")
                print(f"  有名称的记录: {r['name_count']:,}")
        else:
            print("  未找到包含Wiki字段的图层")
        
        print("\n\n推荐分析的图层（按潜在价值排序）:")
        # 按名称记录数排序
        sorted_results = sorted(results, key=lambda x: x['name_count'], reverse=True)
        for i, r in enumerate(sorted_results[:10], 1):
            has_wiki = "*** 有Wiki字段 ***" if r['wiki_fields'] else ""
            print(f"{i}. {r['layer']}: {r['records']:,} 条记录, {r['name_count']:,} 个有名称 {has_wiki}")
        
    except Exception as e:
        print(f"[错误] 处理失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    shapefile_path = r"D:\论文实验\数据集\OSM\jiangxi-251115-free.shp"
    quick_check_all_layers(shapefile_path)


if __name__ == "__main__":
    main()


