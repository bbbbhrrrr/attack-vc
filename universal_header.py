import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch import Tensor
import numpy as np

def generate_universal_header(
    model: nn.Module,
    data_samples: list,
    target_sample: Tensor,
    header_length: int,
    epsilon: float = 0.1,
    lambda_param: float = 1.0,
    n_iters: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tensor:
    """
    生成通用扰动头部
    
    Args:
        model: 目标语音合成模型的说话人编码器
        data_samples: 受害者说话人的语音样本列表
        target_sample: 目标说话人的语音样本
        header_length: 头部长度
        epsilon: 扰动最大幅度
        lambda_param: 损失函数权重
        n_iters: 优化迭代次数
        device: 计算设备
        
    Returns:
        通用扰动头部
    """
    # 初始化通用扰动头部
    header = torch.zeros(data_samples[0].shape[0], data_samples[0].shape[1], header_length, device=device)
    header = header.normal_(0, 0.01).requires_grad_(True)
    
    # 设置优化器
    optimizer = optim.Adam([header], lr=0.001)
    criterion = nn.MSELoss()
    
    # 获取目标说话人的嵌入向量
    with torch.no_grad():
        target_emb = model(target_sample)
    
    # 优化通用扰动头部
    pbar = tqdm(range(n_iters))
    for _ in pbar:
        total_loss = 0
        
        for i, sample in enumerate(data_samples):
            # 截取样本到头部长度
            sample_clip = sample[:, :, :header_length]
            
            # 添加扰动
            perturbed_sample = sample_clip + header * epsilon
            
            # 计算说话人嵌入向量
            with torch.no_grad():
                original_emb = model(sample_clip)
            
            adv_emb = model(perturbed_sample)
            
            # 计算损失
            loss = criterion(adv_emb, target_emb) - lambda_param * criterion(adv_emb, original_emb)
            total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / len(data_samples)
        
        # 优化
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        # 更新进度条
        pbar.set_description(f"Loss: {avg_loss.item():.6f}")
        
        # 约束扰动幅度
        with torch.no_grad():
            header.data = torch.clamp(header.data, -1, 1)
    
    return header * epsilon

def apply_universal_header(audio: Tensor, header: Tensor, epsilon: float = 0.1) -> Tensor:
    """
    将通用扰动头部应用到音频上
    
    Args:
        audio: 输入音频张量 [batch_size, n_mels, time]
        header: 通用扰动头部 [batch_size, n_mels, header_length]
        epsilon: 扰动最大幅度
        
    Returns:
        添加了通用扰动头部的音频
    """
    header_length = header.shape[2]
    
    # 确保头部不超过音频长度
    if header_length > audio.shape[2]:
        header = header[:, :, :audio.shape[2]]
        header_length = header.shape[2]
    
    # 复制音频以避免修改原始数据
    perturbed_audio = audio.clone()
    
    # 应用头部扰动
    perturbed_audio[:, :, :header_length] = audio[:, :, :header_length] + header
    
    return perturbed_audio
