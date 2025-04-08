import os
import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import random

class LibriSpeechDataset(Dataset):
    """LibriSpeech数据集加载器
    
    使用torchaudio加载LibriSpeech数据集，并处理为梅尔频谱图
    
    Args:
        root: 数据集根目录
        split: 数据集划分，如'train-clean-100'、'dev-clean'等
        n_mels: 梅尔滤波器数量
        input_length: 输入序列长度
        output_length: 输出序列长度
        transform: 数据增强转换
    """
    def __init__(self, 
                 root: str, 
                 split: str = "train-clean-100",
                 n_mels: int = 512, 
                 input_length: int = 100,
                 output_length: int = 32,
                 transform = None):
        super().__init__()
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root,
            url=split,
            download=True
        )
        self.n_mels = n_mels
        self.input_length = input_length
        self.output_length = output_length
        self.transform = transform
        
        # 梅尔频谱图提取器
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )
        
        # 归一化器
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 加载音频
        waveform, sample_rate, _, _, _, _ = self.dataset[idx]
        
        # 确保采样率为16kHz
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
            
        # 转换为单声道（如果是立体声）
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 提取梅尔频谱图
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # 归一化到[-1, 1]
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        mel_spec = torch.clamp(mel_spec / 3.0, -1.0, 1.0)
        
        # 确保有足够长度
        if mel_spec.size(2) < self.input_length + self.output_length:
            pad_length = self.input_length + self.output_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
            
        # 随机选择一个起始点
        if mel_spec.size(2) > self.input_length + self.output_length:
            start = random.randint(0, mel_spec.size(2) - self.input_length - self.output_length)
        else:
            start = 0
            
        # 切分为输入和输出部分
        input_mel = mel_spec[:, :, start:start+self.input_length]
        target_mel = mel_spec[:, :, start+self.input_length:start+self.input_length+self.output_length]
        
        # 应用数据增强（如果有的话）
        if self.transform is not None:
            input_mel = self.transform(input_mel)
            
        return input_mel, target_mel

class AudioMelDataset(Dataset):
    """自定义音频数据集加载器
    
    用于加载自定义的音频文件，并处理为梅尔频谱图
    
    Args:
        data_path: 数据集目录
        split: 数据集划分（'train'或'val'）
        n_mels: 梅尔滤波器数量
        input_length: 输入序列长度
        output_length: 输出序列长度
        transform: 数据增强转换
    """
    def __init__(self, 
                 data_path: str, 
                 split: str = "train",
                 n_mels: int = 512, 
                 input_length: int = 100,
                 output_length: int = 32,
                 transform = None):
        super().__init__()
        self.data_path = os.path.join(data_path, split)
        self.n_mels = n_mels
        self.input_length = input_length
        self.output_length = output_length
        self.transform = transform
        
        # 查找所有音频文件
        self.audio_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    self.audio_files.append(os.path.join(root, file))
                    
        print(f"Found {len(self.audio_files)} audio files in {self.data_path}")
        
        # 梅尔频谱图提取器
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )
        
        # 归一化器
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 加载音频
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 确保采样率为16kHz
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
        # 转换为单声道（如果是立体声）
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 提取梅尔频谱图
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # 归一化到[-1, 1]
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        mel_spec = torch.clamp(mel_spec / 3.0, -1.0, 1.0)
        
        # 确保有足够长度
        if mel_spec.size(2) < self.input_length + self.output_length:
            pad_length = self.input_length + self.output_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
            
        # 随机选择一个起始点
        if mel_spec.size(2) > self.input_length + self.output_length:
            start = random.randint(0, mel_spec.size(2) - self.input_length - self.output_length)
        else:
            start = 0
            
        # 切分为输入和输出部分
        input_mel = mel_spec[:, :, start:start+self.input_length]
        target_mel = mel_spec[:, :, start+self.input_length:start+self.input_length+self.output_length]
        
        # 应用数据增强（如果有的话）
        if self.transform is not None:
            input_mel = self.transform(input_mel)
            
        return input_mel, target_mel

# 实用函数
def prepare_mel_tensor(waveform, sample_rate=16000, n_mels=512, normalize=True):
    """将波形转换为梅尔频谱图张量
    
    Args:
        waveform: 输入音频波形
        sample_rate: 采样率
        n_mels: 梅尔滤波器数量
        normalize: 是否归一化
        
    Returns:
        梅尔频谱图张量
    """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # 确保是二维张量 [channels, time]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # 确保采样率为16kHz
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    
    # 提取梅尔频谱图
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    
    # 转换为分贝并归一化
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spec = amplitude_to_db(mel_spec)
    
    if normalize:
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        mel_spec = torch.clamp(mel_spec / 3.0, -1.0, 1.0)
    
    return mel_spec
