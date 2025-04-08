import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Optional

def apply_weighted_constraint(
    perturbation: Tensor,
    sample_rate: int = 16000,
    n_mels: int = 512,
    epsilon: float = 0.1,
    low_freq_weight: float = 1.15,
    mid_freq_weight: float = 0.85,
    high_freq_weight: float = 1.0
) -> Tensor:
    """
    应用基于权重的扰动约束
    
    根据论文附录B中描述，人类对中频段(1.6kHz-4kHz)最为敏感，
    因此减少这部分的扰动幅度，同时在低频和高频区域增加扰动强度以补偿保护效果。
    论文研究表明，超过60%的受测者认为使用权重约束后的扰动几乎无感知，而未使用权重
    约束时，只有不到50%的人认为扰动可以忽略。
    
    Args:
        perturbation: 原始扰动张量 [batch_size, n_mels, time]
        sample_rate: 采样率
        n_mels: 梅尔滤波器数量
        epsilon: 扰动最大幅度
        low_freq_weight: 低频段权重系数 (< 1.6kHz)，论文中为1.15
        mid_freq_weight: 中频段权重系数 (1.6kHz-4kHz)，论文中为0.85
        high_freq_weight: 高频段权重系数 (> 4kHz)，论文中为1.0
        
    Returns:
        应用了权重约束的扰动，有效降低人耳对扰动的感知
    """
    # 计算频率到mel bin的映射
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_bin = lambda mel: int(n_mels * mel / hz_to_mel(sample_rate / 2))
    
    # 计算各个频段对应的bin索引
    low_freq_threshold = 1600  # Hz，论文中指定的低频阈值
    high_freq_threshold = 4000  # Hz，论文中指定的高频阈值
    
    low_freq_bin = mel_to_bin(hz_to_mel(low_freq_threshold))
    high_freq_bin = mel_to_bin(hz_to_mel(high_freq_threshold))
    
    # 复制扰动以避免修改原始数据
    weighted_perturbation = perturbation.clone()
    
    # 应用不同频段的权重 - 论文公式(5)的实现
    # 根据附录B，中频区域(1.6kHz-4kHz)使用较低权重，低频和高频区域使用较高权重
    # 这样可以显著降低人耳对噪声的感知
    weighted_perturbation[:, :low_freq_bin, :] *= low_freq_weight
    weighted_perturbation[:, low_freq_bin:high_freq_bin, :] *= mid_freq_weight
    weighted_perturbation[:, high_freq_bin:, :] *= high_freq_weight
    
    # 确保扰动幅度不超过epsilon
    weighted_perturbation = torch.clamp(weighted_perturbation, -epsilon, epsilon)
    
    return weighted_perturbation

def visualize_weight_distribution(
    sample_rate: int = 16000, 
    n_mels: int = 512,
    low_freq_weight: float = 1.15,
    mid_freq_weight: float = 0.85,
    high_freq_weight: float = 1.0
) -> None:
    """
    可视化权重分布图
    
    生成类似论文图10的权重分布可视化图，显示不同频率段的权重
    
    Args:
        sample_rate: 采样率
        n_mels: 梅尔滤波器数量
        low_freq_weight: 低频段权重系数 (< 1.6kHz)
        mid_freq_weight: 中频段权重系数 (1.6kHz-4kHz)
        high_freq_weight: 高频段权重系数 (> 4kHz)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装matplotlib: pip install matplotlib")
        return
    
    # 计算频率到mel bin的映射
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_bin = lambda mel: int(n_mels * mel / hz_to_mel(sample_rate / 2))
    bin_to_hz = lambda bin: 700 * (10**(bin * hz_to_mel(sample_rate / 2) / n_mels / 2595) - 1)
    
    # 计算各个频段对应的bin索引
    low_freq_threshold = 1600  # Hz
    high_freq_threshold = 4000  # Hz
    
    low_freq_bin = mel_to_bin(hz_to_mel(low_freq_threshold))
    high_freq_bin = mel_to_bin(hz_to_mel(high_freq_threshold))
    
    # 创建频率和权重数组
    bins = np.arange(n_mels)
    freqs = np.array([bin_to_hz(b) for b in bins])
    weights = np.ones(n_mels)
    
    weights[:low_freq_bin] = low_freq_weight
    weights[low_freq_bin:high_freq_bin] = mid_freq_weight
    weights[high_freq_bin:] = high_freq_weight
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, weights)
    plt.axvline(x=low_freq_threshold, color='r', linestyle='--', label='1600 Hz')
    plt.axvline(x=high_freq_threshold, color='g', linestyle='--', label='4000 Hz')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('权重')
    plt.title('VSMask权重约束分布')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('weight_distribution.png')
    plt.close()
    print("权重分布图已保存为weight_distribution.png")

if __name__ == "__main__":
    # 生成与论文图10类似的权重分布图
    visualize_weight_distribution()
    
    # 测试权重约束对扰动的影响
    print("测试权重约束对扰动的影响...")
    dummy_perturbation = torch.randn(1, 512, 100)
    weighted_perturbation = apply_weighted_constraint(dummy_perturbation)
    
    print(f"扰动形状: {dummy_perturbation.shape}")
    print(f"加权后扰动形状: {weighted_perturbation.shape}")
    print(f"低频区域平均幅度: {weighted_perturbation[0, :mel_to_bin(hz_to_mel(1600)), :].abs().mean().item():.4f}")
    print(f"中频区域平均幅度: {weighted_perturbation[0, mel_to_bin(hz_to_mel(1600)):mel_to_bin(hz_to_mel(4000)), :].abs().mean().item():.4f}")
    print(f"高频区域平均幅度: {weighted_perturbation[0, mel_to_bin(hz_to_mel(4000)):, :].abs().mean().item():.4f}")