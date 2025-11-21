"""
OSM Shapefile字段属性分析工具
详细展示shapefile中的所有字段信息，帮助识别wiki相关字段
"""

import geopandas as gpd
import pandas as pd
from collections import Counter
import fiona
import sys
import io

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def list_shapefile_layers(shapefile_path: str):
    """
    列出shapefile中的所有图层
    
    Args:
        shapefile_path: shapefile文件路径
        
    Returns:
        list: 图层名称列表
    """
    try:
        layers = fiona.listlayers(shapefile_path)
        return layers
    except Exception as e:
        print(f"无法读取图层列表: {e}")
        return []


def analyze_layer_properties(shapefile_path: str, layer_name: str = None):
    """
    详细分析shapefile某个图层的所有字段属性
    
    Args:
        shapefile_path: shapefile文件路径
        layer_name: 图层名称，如果为None则使用默认图层
    """
    layer_info = f" (图层: {layer_name})" if layer_name else ""
    print("=" * 80)
    print(f"OSM Shapefile 字段属性详细分析{layer_info}")
    print("=" * 80)
    print(f"\n文件路径: {shapefile_path}\n")
    
    try:
        # 读取shapefile
        print(f"正在读取shapefile文件{layer_info}...")
        if layer_name:
            gdf = gpd.read_file(shapefile_path, layer=layer_name)
        else:
            gdf = gpd.read_file(shapefile_path)
        print(f"[OK] 成功读取，共 {len(gdf)} 条记录\n")
        
        # ===== 1. 显示所有字段名称 =====
        print("=" * 80)
        print("1. 所有字段列表")
        print("=" * 80)
        all_columns = list(gdf.columns)
        for i, col in enumerate(all_columns, 1):
            print(f"  {i:2d}. {col}")
        print()
        
        # ===== 2. 详细字段信息 =====
        print("=" * 80)
        print("2. 详细字段信息（数据类型、非空值统计）")
        print("=" * 80)
        print(f"{'序号':<4} {'字段名':<20} {'数据类型':<15} {'非空值数量':<12} {'非空比例':<10} {'唯一值数量':<12}")
        print("-" * 80)
        
        for i, col in enumerate(gdf.columns, 1):
            if col == 'geometry':
                continue
            
            dtype = str(gdf[col].dtype)
            non_null_count = gdf[col].notna().sum()
            non_null_ratio = non_null_count / len(gdf) * 100
            unique_count = gdf[col].nunique()
            
            print(f"{i:<4} {col:<20} {dtype:<15} {non_null_count:<12} {non_null_ratio:>6.2f}%    {unique_count:<12}")
        print()
        
        # ===== 3. 查找可能的wiki相关字段 =====
        print("=" * 80)
        print("3. 查找可能包含Wiki信息的字段")
        print("=" * 80)
        
        # 搜索关键词
        wiki_keywords = ['wiki', 'wikidata', 'wikipedia', 'Wiki', 'WIKI', 
                        'url', 'URL', 'link', 'LINK', 'web', 'WEB',
                        'website', 'WEBSITE', 'ref', 'REF', 'source', 'SOURCE']
        
        potential_wiki_fields = []
        for col in gdf.columns:
            if col == 'geometry':
                continue
            for keyword in wiki_keywords:
                if keyword in col or keyword in str(col).lower():
                    potential_wiki_fields.append(col)
                    break
        
        if potential_wiki_fields:
            print(f"找到 {len(potential_wiki_fields)} 个可能包含Wiki信息的字段:")
            for col in potential_wiki_fields:
                non_null = gdf[col].notna().sum()
                print(f"  * {col} - {non_null} 条非空记录")
        else:
            print("[警告] 未找到明显的Wiki相关字段名")
        print()
        
        # ===== 4. 显示每个字段的样本数据 =====
        print("=" * 80)
        print("4. 每个字段的样本数据（前10个非空值）")
        print("=" * 80)
        
        for col in gdf.columns:
            if col == 'geometry':
                continue
            
            print(f"\n【字段: {col}】")
            print(f"  数据类型: {gdf[col].dtype}")
            
            # 获取非空值
            non_null_values = gdf[col].dropna()
            
            if len(non_null_values) == 0:
                print("  [警告] 所有值都为空")
                continue
            
            print(f"  非空值数量: {len(non_null_values)} / {len(gdf)} ({len(non_null_values)/len(gdf)*100:.1f}%)")
            print(f"  唯一值数量: {gdf[col].nunique()}")
            
            # 显示前10个非空样本
            print("  样本数据 (前10个非空值):")
            sample_values = non_null_values.head(10).tolist()
            for i, val in enumerate(sample_values, 1):
                # 限制显示长度
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"    {i:2d}. {val_str}")
            
            # 如果是字符串类型，显示值的长度分布
            if gdf[col].dtype == 'object':
                lengths = non_null_values.astype(str).str.len()
                print(f"  值长度统计: 最小={lengths.min()}, 最大={lengths.max()}, 平均={lengths.mean():.1f}")
                
                # 检查是否可能是URL
                url_like = non_null_values.astype(str).str.contains(r'http|www|://|\.com|\.org', case=False, na=False).sum()
                if url_like > 0:
                    print(f"  [重要] 可能包含URL: {url_like} 条记录")
        
        print("\n" + "=" * 80)
        
        # ===== 5. 查找包含特定模式的数据 =====
        print("5. 深度检查：查找可能被忽略的Wiki链接")
        print("=" * 80)
        
        wiki_patterns = ['Q[0-9]+', 'wiki', 'wikipedia.org', 'wikidata.org', 
                        'dbpedia', 'en.wikipedia', 'zh.wikipedia']
        
        found_wiki_data = False
        for col in gdf.columns:
            if col == 'geometry':
                continue
            
            # 检查字符串类型字段
            if gdf[col].dtype == 'object':
                for pattern in wiki_patterns:
                    matches = gdf[col].astype(str).str.contains(pattern, case=False, na=False, regex=True).sum()
                    if matches > 0:
                        found_wiki_data = True
                        print(f"\n  [发现] 在字段 '{col}' 中找到 {matches} 条包含 '{pattern}' 的记录")
                        
                        # 显示样本
                        sample = gdf[gdf[col].astype(str).str.contains(pattern, case=False, na=False)][col].head(3)
                        print(f"    样本:")
                        for idx, val in enumerate(sample, 1):
                            val_str = str(val)
                            if len(val_str) > 80:
                                val_str = val_str[:80] + "..."
                            print(f"      {idx}. {val_str}")
        
        if not found_wiki_data:
            print("\n  [警告] 未在任何字段中找到Wiki相关数据")
        
        print("\n" + "=" * 80)
        
        # ===== 6. 几何信息 =====
        print("6. 几何信息")
        print("=" * 80)
        geom_types = gdf.geometry.geom_type.value_counts()
        print("几何类型分布:")
        for geom_type, count in geom_types.items():
            print(f"  * {geom_type}: {count} ({count/len(gdf)*100:.1f}%)")
        
        # 坐标系统
        print(f"\n坐标参考系统 (CRS): {gdf.crs}")
        
        # 边界框
        bounds = gdf.total_bounds
        print(f"边界框: ")
        print(f"  经度范围: {bounds[0]:.6f} ~ {bounds[2]:.6f}")
        print(f"  纬度范围: {bounds[1]:.6f} ~ {bounds[3]:.6f}")
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        
        # ===== 7. 保存字段信息到文件 =====
        print("\n正在保存字段信息到CSV文件...")
        field_info = []
        for col in gdf.columns:
            if col == 'geometry':
                continue
            field_info.append({
                '字段名': col,
                '数据类型': str(gdf[col].dtype),
                '非空值数量': gdf[col].notna().sum(),
                '非空比例': f"{gdf[col].notna().sum() / len(gdf) * 100:.2f}%",
                '唯一值数量': gdf[col].nunique(),
                '样本值': str(gdf[col].dropna().head(1).tolist()[0]) if len(gdf[col].dropna()) > 0 else 'N/A'
            })
        
        df_info = pd.DataFrame(field_info)
        if layer_name:
            output_path = shapefile_path.replace('.shp', f'_field_analysis_{layer_name}.csv')
        else:
            output_path = shapefile_path.replace('.shp', '_field_analysis.csv')
        df_info.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[OK] 字段信息已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[错误] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    shapefile_path = r"D:\论文实验\数据集\OSM\jiangxi-251115-free.shp"
    
    # 首先列出所有图层
    print("=" * 80)
    print("检测Shapefile中的所有图层")
    print("=" * 80)
    layers = list_shapefile_layers(shapefile_path)
    
    if layers:
        print(f"\n找到 {len(layers)} 个图层:")
        for i, layer in enumerate(layers, 1):
            print(f"  {i}. {layer}")
        print()
        
        # 分析每个图层
        for i, layer in enumerate(layers, 1):
            print("\n" + "=" * 80)
            print(f"分析图层 {i}/{len(layers)}: {layer}")
            print("=" * 80)
            success = analyze_layer_properties(shapefile_path, layer)
            
            if i < len(layers):
                print("\n按Enter键继续分析下一个图层，或输入'q'退出...")
                # 自动继续，不等待输入（方便批量分析）
                # user_input = input()
                # if user_input.lower() == 'q':
                #     break
    else:
        print("未找到图层，尝试分析默认图层...")
        analyze_layer_properties(shapefile_path)


if __name__ == "__main__":
    main()

