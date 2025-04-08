import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple, Union

class DownSampleBlock(nn.Module):
    """下采样块
    
    使用卷积进行下采样,包含反射填充、卷积、批归一化和PReLU激活
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int]=(3, 3), 
                 stride: Tuple[int, int]=(1, 2)):
        super(DownSampleBlock, self).__init__()
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))  # 边缘反射填充
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)       # 填充
        x = self.conv(x)      # 卷积下采样
        x = self.norm(x)      # 批归一化
        x = self.act(x)       # 激活函数
        return x

class UpSampleBlock(nn.Module):
    """上采样块
    
    使用转置卷积进行上采样,包含转置卷积和LeakyReLU激活
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数 
        kernel_size: 卷积核大小
        stride: 步长
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int]=(3, 3),
                 stride: Tuple[int, int]=(2, 2)):
        super(UpSampleBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1, 1),
            output_padding=(0, 0)
        )
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_transpose(x)  # 转置卷积上采样
        x = self.act(x)             # 激活函数
        return x

class PredictiveModel(nn.Module):
    """预测模型
    
    根据论文附录A和表4构建VSMask实时预测模型。论文指出通过测试不同的
    内核大小、步长和层数，确定了(3,3)内核大小能够在保护性能和模型复杂度之间
    达到最佳平衡。最终使用tanh激活层归一化预测扰动。
    
    模型结构完全遵循表4中针对AdaIN-VC模型的详细网络参数配置，包含7个
    下采样层和5个上采样层，以保证最佳的预测性能。
    
    Args:
        n_mels: 梅尔频谱的频率维度
        input_length: 输入的时间维度长度
        output_length: 输出的时间维度长度
    """
    def __init__(self, n_mels: int=512, input_length: int=100, output_length: int=32):
        super(PredictiveModel, self).__init__()
        self.n_mels = n_mels
        self.input_length = input_length
        self.output_length = output_length
        
        # 保存输入参数
        self.model_config = {
            "n_mels": n_mels,
            "input_length": input_length,
            "output_length": output_length
        }
        
        # 构建7个下采样块，严格按照论文表4配置
        # 按表4所示: Input Size → Kernel Size → Stride
        self.down_blocks = nn.ModuleList([
            DownSampleBlock(1, 32, kernel_size=(3, 3), stride=(1, 2)),     # 1×512×100 → 32×512×50
            DownSampleBlock(32, 64, kernel_size=(3, 3), stride=(2, 2)),    # 32×512×50 → 64×256×25
            DownSampleBlock(64, 128, kernel_size=(3, 3), stride=(2, 2)),   # 64×256×25 → 128×128×13
            DownSampleBlock(128, 256, kernel_size=(3, 3), stride=(2, 2)),  # 128×128×13 → 256×64×7
            DownSampleBlock(256, 256, kernel_size=(3, 3), stride=(2, 2)),  # 256×64×7 → 256×32×4
            DownSampleBlock(256, 512, kernel_size=(3, 3), stride=(2, 2)),  # 256×32×4 → 512×16×2
            DownSampleBlock(512, 512, kernel_size=(3, 3), stride=(2, 2)),  # 512×16×2 → 512×8×1
        ])
        
        # 构建5个上采样块，严格按照论文表4配置
        self.up_blocks = nn.ModuleList([
            UpSampleBlock(512, 512, kernel_size=(3, 3), stride=(2, 2)),    # 512×8×1 → 512×16×2
            UpSampleBlock(512, 256, kernel_size=(3, 3), stride=(2, 2)),    # 512×16×2 → 256×32×4
            UpSampleBlock(256, 128, kernel_size=(3, 3), stride=(2, 2)),    # 256×32×4 → 128×64×8
            UpSampleBlock(128, 64, kernel_size=(3, 3), stride=(2, 2)),     # 128×64×8 → 64×128×16
            UpSampleBlock(64, 32, kernel_size=(3, 3), stride=(2, 2)),      # 64×128×16 → 32×256×16
        ])
        
        # 输出层，根据表4中最后一行的配置，输出大小应为1×512×32
        self.output_layer = nn.Conv2d(32, 1, kernel_size=(1, 1))
        # 论文指出使用tanh激活层归一化预测扰动
        self.tanh = nn.Tanh()  
        
    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        按照论文附录A中的描述，模型接收和输出2D向量，具有相同的维度但不同的长度。
        经过下采样和上采样处理后，输出为扰动预测。
        
        Args:
            x: 输入张量 [batch_size, 1, n_mels, input_length]
            
        Returns:
            输出张量 [batch_size, 1, n_mels, output_length]
        """
        # 下采样路径
        features = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features.append(x)
        
        # 上采样路径(包含跳跃连接)
        for i, block in enumerate(self.up_blocks):
            x = block(x)
            # 添加跳跃连接以提高模型性能
            if i < len(features) - 1:  # 跳过最深层特征
                skip = features[-(i+2)]  # 从倒数第二个特征开始
                if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # 确保输出尺寸与表4中指定的一致 (1×512×32)
        if x.size(2) != self.n_mels or x.size(3) != self.output_length:
            x = F.interpolate(x, size=(self.n_mels, self.output_length), mode='bilinear', align_corners=False)
        
        # 输出层
        x = self.output_layer(x)
        # 使用tanh归一化扰动输出，限制扰动范围在[-1,1]，如论文所述
        x = self.tanh(x)  
        return x
    
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.model_config
        }, path)
        
    @classmethod
    def load_model(cls, path: str) -> 'PredictiveModel':
        """加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的预测模型
        """
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

def build_predictive_model(n_mels: int=512, input_length: int=100, output_length: int=32) -> PredictiveModel:
    """构建预测模型
    
    按照论文表4中的规格构建VSMask实时预测模型。论文指出(3,3)内核大小提供了
    保护性能和模型复杂度之间的最佳权衡。
    
    Args:
        n_mels: 梅尔频谱的频率维度，论文中为512
        input_length: 输入的时间维度长度，论文中为100
        output_length: 输出的时间维度长度，论文中为32
        
    Returns:
        构建的预测模型，符合表4中的架构配置
    """
    return PredictiveModel(
        n_mels=n_mels,
        input_length=input_length, 
        output_length=output_length
    )
