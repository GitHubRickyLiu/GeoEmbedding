# 空间对比学习损失实现说明

## 概述

为GeoLM项目增加了空间对比学习损失功能，使得地理实体在语义空间中的距离与现实世界的地理距离相近。

## 修改内容

### 1. 模型修改 (`src/models/spatial_bert_model.py`)

#### 新增导入
```python
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import torch.nn.functional as F
```

#### 新增损失函数类

##### `SpatialContrastiveLoss`
- **功能**: 实现基于InfoNCE的空间对比学习损失
- **参数**:
  - `temperature`: 温度参数，默认0.1
  - `distance_threshold`: 空间距离阈值，用于区分正负样本，默认1000
- **逻辑**:
  - 计算所有实体对之间的欧几里得空间距离
  - 空间距离 < 阈值：正样本对（应相似）
  - 空间距离 >= 阈值：负样本对（应不相似）
  - 使用InfoNCE损失优化

##### `SpatialTripletLoss`
- **功能**: 实现基于三元组的空间对比学习损失
- **参数**:
  - `margin`: 三元组损失的边距，默认1.0
  - `distance_threshold`: 空间距离阈值，默认1000
- **逻辑**:
  - 对于空间距离近的实体对作为正样本
  - 对于空间距离远的实体对作为负样本
  - 使用三元组损失：`max(0, dist(anchor,positive) - dist(anchor,negative) + margin)`

#### 修改 `SpatialBertForMaskedLM.forward()`
- 新增参数: `return_geo_nl_embeddings=False`
- 当 `return_geo_nl_embeddings=True` 时，返回geo和nl的实体嵌入

### 2. 训练脚本修改 (`src/train_joint.py`)

#### 新增导入
```python
from models.spatial_bert_model import SpatialContrastiveLoss, SpatialTripletLoss
```

#### 训练流程修改
1. **初始化空间对比损失**:
```python
spatial_contrastive_criterion = SpatialContrastiveLoss(temperature=0.1, distance_threshold=1000)
```

2. **计算空间损失**:
```python
# 获取每个实体的拼接嵌入（geo + nl）
geo_embeddings = outputs1.hidden_states  # [batch_size, hidden_size]
nl_embeddings = outputs2.hidden_states   # [batch_size, hidden_size]
concat_embeddings = torch.cat([geo_embeddings, nl_embeddings], dim=-1)  # [batch_size, 2*hidden_size]

# 获取每个实体的经纬度坐标
geo_coordinates = torch.stack([
    geo_position_list_x[:, 0],  # 实体经度
    geo_position_list_y[:, 0]   # 实体纬度
], dim=1).to(device)  # [batch_size, 2]

# 计算空间对比学习损失
loss4 = spatial_contrastive_criterion(concat_embeddings, geo_coordinates)
```

3. **总损失计算**:
```python
loss = loss1 + loss3 + loss4  # MLM损失 + 对比损失 + 空间损失
```

4. **进度显示更新**:
```python
loop.set_postfix({'loss':loss.item(),'mlm':loss1.item(),'contrast':loss3.item(),'spatial':loss4.item()})
```

## 数据流程

1. **输入数据**:
   - `geo_data`: 地理实体数据（包含经纬度坐标）
   - `nl_data`: 自然语言实体数据
   - `concat_data`: 拼接数据

2. **嵌入生成**:
   - `geo_embeddings`: 地理数据的实体嵌入
   - `nl_embeddings`: 自然语言数据的实体嵌入
   - `concat_embeddings`: 拼接后的嵌入 `[geo_emb, nl_emb]`

3. **空间信息**:
   - `geo_coordinates`: 实体的经纬度坐标 `[lng, lat]`

4. **损失计算**:
   - 空间距离近的实体：嵌入应相似（正样本）
   - 空间距离远的实体：嵌入应不相似（负样本）

## 超参数建议

- `temperature`: 0.1 (InfoNCE温度参数)
- `distance_threshold`: 1000 (空间距离阈值，单位与数据归一化相关)
- 损失权重: 与现有损失保持相同权重 (1.0)

## 测试

运行测试脚本验证实现：
```bash
python test_spatial_loss.py
```

## 注意事项

1. **坐标归一化**: 确保经纬度坐标与 `distance_norm_factor` 参数一致
2. **批次大小**: 空间损失计算复杂度为 O(batch_size²)，较大批次可能影响训练速度
3. **距离阈值**: 根据数据集的地理分布调整 `distance_threshold`
4. **GPU内存**: 拼接嵌入维度翻倍，需要确保GPU内存充足

## 未来优化

1. **效率优化**: 考虑使用近似最近邻搜索优化大批量计算
2. **自适应阈值**: 根据批次数据动态调整距离阈值
3. **多尺度损失**: 考虑不同空间尺度下的对比学习
