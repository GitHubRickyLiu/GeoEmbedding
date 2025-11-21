#!/usr/bin/env python3
"""
测试空间对比学习损失函数
"""
import torch
import torch.nn as nn
import sys
sys.path.append('src')

from models.spatial_bert_model import SpatialContrastiveLoss, SpatialTripletLoss

def test_spatial_contrastive_loss():
    print("Testing SpatialContrastiveLoss...")

    # 创建损失函数
    loss_fn = SpatialContrastiveLoss(temperature=0.1, distance_threshold=1000)

    # 创建测试数据
    batch_size = 4
    hidden_size = 768

    # 模拟拼接嵌入 [batch_size, 2*hidden_size]
    embeddings = torch.randn(batch_size, 2*hidden_size)

    # 模拟经纬度坐标 [batch_size, 2] - [lng, lat]
    # 实体0: (0, 0)
    # 实体1: (1, 1) - 距离实体0约1.41
    # 实体2: (1000, 1000) - 距离实体0很远
    # 实体3: (2, 2) - 距离实体0约2.82
    coordinates = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [1000.0, 1000.0],
        [2.0, 2.0]
    ], dtype=torch.float32)

    # 计算损失
    loss = loss_fn(embeddings, coordinates)
    print(f"Loss: {loss.item():.4f}")

    # 验证损失不为NaN
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss >= 0, "Loss should be non-negative"

    print("SpatialContrastiveLoss test passed!")

def test_spatial_triplet_loss():
    print("\nTesting SpatialTripletLoss...")

    # 创建损失函数
    loss_fn = SpatialTripletLoss(margin=1.0, distance_threshold=1000)

    # 创建测试数据
    batch_size = 4
    hidden_size = 768

    # 模拟拼接嵌入
    embeddings = torch.randn(batch_size, 2*hidden_size)

    # 模拟经纬度坐标
    coordinates = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [1000.0, 1000.0],
        [2.0, 2.0]
    ], dtype=torch.float32)

    # 计算损失
    loss = loss_fn(embeddings, coordinates)
    print(f"Loss: {loss.item():.4f}")

    # 验证损失不为NaN
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss >= 0, "Loss should be non-negative"

    print("SpatialTripletLoss test passed!")

if __name__ == "__main__":
    test_spatial_contrastive_loss()
    test_spatial_triplet_loss()
    print("\nAll tests passed!")
