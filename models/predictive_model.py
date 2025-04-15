import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
                stride: Tuple[int, int]):
        """
        下采样模块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
        """
        super(DownSamplingBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d((kernel_size[0]//2, kernel_size[0]//2, 
                               kernel_size[1]//2, kernel_size[1]//2)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
                stride: Tuple[int, int]):
        """
        上采样模块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
        """
        super(UpSamplingBlock, self).__init__()
        
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)

class PredictiveModel(nn.Module):
    def __init__(self, mel_bins: int = 80, time_dim: int = 100):
        """
        VSMask预测模型
        
        Args:
            mel_bins: 梅尔频谱图的频率维度
            time_dim: 输入时间维度
        """
        super(PredictiveModel, self).__init__()
        
        # 定义下采样网络结构
        self.down_blocks = nn.ModuleList([
            DownSamplingBlock(1, 32, (3, 3), (1, 2)),
            DownSamplingBlock(32, 64, (3, 3), (2, 2)),
            DownSamplingBlock(64, 128, (3, 3), (2, 2)),
            DownSamplingBlock(128, 256, (3, 3), (2, 2)),
            DownSamplingBlock(256, 256, (3, 3), (2, 2)),
            DownSamplingBlock(256, 512, (3, 3), (2, 2)),
            DownSamplingBlock(512, 512, (3, 3), (2, 2))
        ])
        
        # 定义上采样网络结构
        self.up_blocks = nn.ModuleList([
            UpSamplingBlock(512, 256, (3, 3), (2, 2)),
            UpSamplingBlock(256, 128, (3, 3), (2, 2)),
            UpSamplingBlock(128, 64, (3, 3), (2, 2)),
            UpSamplingBlock(64, 32, (3, 3), (2, 2)),
            UpSamplingBlock(32, 1, (3, 3), (2, 2))
        ])
        
        # 用于最终输出的tanh激活
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入梅尔频谱图 [B, 1, F, T]
            
        Returns:
            预测的扰动 [B, 1, F, T']
        """
        # 下采样部分
        down_features = []
        for block in self.down_blocks:
            x = block(x)
            down_features.append(x)
        
        # 上采样部分
        for i, block in enumerate(self.up_blocks):
            x = block(x)
        
        # 应用tanh激活函数归一化输出
        x = self.tanh(x)
        
        return x
