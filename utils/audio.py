import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional

class MelSpectrogramConverter:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        """
        梅尔频谱转换器
        
        Args:
            sample_rate: 音频采样率
            n_fft: FFT大小
            hop_length: 帧移
            n_mels: 梅尔滤波器组数量
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # 用于逆变换的近似变换（需要外部vocoder进行精确重建）
        self.inverse_mel_transform = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        将波形转换为梅尔频谱图
        
        Args:
            waveform: 输入波形 [1, T]
            
        Returns:
            梅尔频谱图 [1, n_mels, T']
        """
        # 应用对数梅尔频谱变换
        mel_spec = self.mel_transform(waveform)
        # 转换为对数刻度
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        return log_mel_spec
    
    def mel_to_waveform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        将梅尔频谱图转换回波形（近似重建）
        
        Args:
            mel_spec: 梅尔频谱图 [1, n_mels, T]
            
        Returns:
            重建的波形 [1, T']
        """
        # 从对数刻度转换回线性刻度
        linear_mel_spec = torch.pow(10, mel_spec)
        # 转换回线性频谱
        spec = self.inverse_mel_transform(linear_mel_spec)
        # 使用Griffin-Lim算法重建波形
        waveform = self.griffin_lim(spec)
        return waveform.unsqueeze(0)
    
    def apply_weighted_constraint(self, perturbation: torch.Tensor, 
                                 epsilon1: float = 0.1, 
                                 epsilon2: float = 0.05, 
                                 epsilon3: float = 0.08) -> torch.Tensor:
        """
        对扰动应用基于权重的约束
        
        Args:
            perturbation: 梅尔频谱图扰动
            epsilon1: 低频约束值
            epsilon2: 中频约束值
            epsilon3: 高频约束值
            
        Returns:
            应用了基于权重约束的扰动
        """
        # 获取梅尔频谱图的频率维度
        _, freq_dim, _ = perturbation.shape
        
        # 定义频率范围
        low_freq_end = int(freq_dim * 0.3)  # 低频范围（对应约1.6kHz以下）
        high_freq_start = int(freq_dim * 0.7)  # 高频范围（对应约4kHz以上）
        
        # 分割扰动为低、中、高频部分
        low_freq = perturbation[:, :low_freq_end, :]
        mid_freq = perturbation[:, low_freq_end:high_freq_start, :]
        high_freq = perturbation[:, high_freq_start:, :]
        
        # 应用不同的约束
        low_freq_constrained = torch.clamp(low_freq, -epsilon1, epsilon1)
        mid_freq_constrained = torch.clamp(mid_freq, -epsilon2, epsilon2)
        high_freq_constrained = torch.clamp(high_freq, -epsilon3, epsilon3)
        
        # 重组扰动
        weighted_perturbation = torch.cat(
            [low_freq_constrained, mid_freq_constrained, high_freq_constrained],
            dim=1
        )
        
        return weighted_perturbation

def apply_random_shift(waveform: torch.Tensor, max_shift: int = 100) -> torch.Tensor:
    """
    对波形应用随机移位
    
    Args:
        waveform: 输入波形 [1, T]
        max_shift: 最大移位量
        
    Returns:
        移位后的波形 [1, T]
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    
    if shift > 0:
        # 右移
        shifted_waveform = torch.cat([
            torch.zeros(1, shift, device=waveform.device),
            waveform[:, :-shift]
        ], dim=1)
    elif shift < 0:
        # 左移
        shifted_waveform = torch.cat([
            waveform[:, -shift:],
            torch.zeros(1, -shift, device=waveform.device)
        ], dim=1)
    else:
        shifted_waveform = waveform
    
    return shifted_waveform
