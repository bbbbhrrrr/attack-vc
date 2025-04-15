import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class UniversalPerturbationHeader:
    def __init__(self, mel_bins: int = 80, time_length: int = 100, device: str = 'cuda'):
        """
        通用扰动头模型
        
        Args:
            mel_bins: 梅尔频谱图的频率维度
            time_length: 时间维度长度
            device: 使用的设备
        """
        self.mel_bins = mel_bins
        self.time_length = time_length
        self.device = device
        
        # 初始化通用扰动头
        self.header = torch.zeros((1, 1, mel_bins, time_length), device=device)
        self.header.requires_grad = True
    
    def optimize(self, source_mel: torch.Tensor, target_mel: torch.Tensor, 
                speaker_encoder, optimizer, num_iterations: int = 1000, 
                epsilon: float = 0.1, lambda_param: float = 0.5) -> None:
        """
        优化通用扰动头
        
        Args:
            source_mel: 源说话人梅尔频谱图 [B, 1, F, T]
            target_mel: 目标说话人梅尔频谱图 [B, 1, F, T]
            speaker_encoder: 说话人编码器模型
            optimizer: 优化器
            num_iterations: 优化迭代次数
            epsilon: 扰动最大幅度
            lambda_param: 损失函数中的平衡参数
        """
        for i in range(num_iterations):
            # 添加扰动
            perturbed_mel = source_mel + self.header
            
            # 约束扰动幅度
            perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)
            
            # 计算说话人嵌入
            source_embedding = speaker_encoder(source_mel)
            target_embedding = speaker_encoder(target_mel)
            perturbed_embedding = speaker_encoder(perturbed_mel)
            
            # 计算损失：最小化与目标说话人的距离，最大化与源说话人的距离
            loss_target = F.mse_loss(perturbed_embedding, target_embedding)
            loss_source = F.mse_loss(perturbed_embedding, source_embedding)
            
            loss = loss_target - lambda_param * loss_source
            
            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 约束扰动幅度
            with torch.no_grad():
                self.header.data = torch.clamp(self.header.data, -epsilon, epsilon)
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    def apply_header(self, source_mel: torch.Tensor) -> torch.Tensor:
        """
        应用通用扰动头到输入梅尔频谱图
        
        Args:
            source_mel: 源说话人梅尔频谱图 [B, 1, F, T]
            
        Returns:
            添加了扰动头的梅尔频谱图 [B, 1, F, T]
        """
        # 确保扰动头与输入形状兼容
        _, _, F, T = source_mel.shape
        header = self.header
        
        # 如果输入比扰动头短，则裁剪扰动头
        if T < self.time_length:
            header = header[:, :, :, :T]
        
        # 如果输入比扰动头长，则仅在开头添加扰动
        perturbed_mel = source_mel.clone()
        perturbed_mel[:, :, :, :header.shape[3]] += header
        
        # 约束结果范围
        perturbed_mel = torch.clamp(perturbed_mel, -1.0, 1.0)
        
        return perturbed_mel
    
    def save(self, path: str) -> None:
        """保存通用扰动头到文件"""
        torch.save(self.header, path)
    
    def load(self, path: str) -> None:
        """从文件加载通用扰动头"""
        self.header = torch.load(path, map_location=self.device)
        self.header.requires_grad = True
