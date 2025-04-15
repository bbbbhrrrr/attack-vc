import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple, Dict, Optional

class VCTKDataset(Dataset):
    def __init__(self, root_dir: str, speaker_id: str, transform=None, 
                 split='train', window_size=100, shift_size=50, sample_rate=16000):
        """
        VCTK数据集加载类
        
        Args:
            root_dir: 数据集根目录
            speaker_id: 用于训练的说话人ID
            transform: 数据转换函数
            split: 'train'或'test'
            window_size: 滑动窗口大小
            shift_size: 滑动窗口移动步长
            sample_rate: 音频采样率
        """
        self.root_dir = root_dir
        self.speaker_id = speaker_id
        self.transform = transform
        self.window_size = window_size
        self.shift_size = shift_size
        self.sample_rate = sample_rate
        
        # 获取说话人的所有音频文件
        speaker_dir = os.path.join(root_dir, f'p{speaker_id}')
        self.audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
        
        # 划分训练集和测试集
        random.seed(42)  # 固定随机种子确保可复现性
        random.shuffle(self.audio_files)
        
        if split == 'train':
            self.audio_files = self.audio_files[:int(0.8 * len(self.audio_files))]
        else:
            self.audio_files = self.audio_files[int(0.8 * len(self.audio_files)):]
        
        # 预处理音频并创建滑动窗口片段
        self.segments = self._preprocess_audio()
    
    def _preprocess_audio(self) -> List[Tuple[torch.Tensor, int]]:
        """预处理音频并创建滑动窗口片段"""
        segments = []
        
        for audio_file in self.audio_files:
            file_path = os.path.join(self.root_dir, f'p{self.speaker_id}', audio_file)
            waveform, sr = torchaudio.load(file_path)
            
            # 重采样到目标采样率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 应用额外的转换（如果有）
            if self.transform:
                waveform = self.transform(waveform)
            
            # 创建滑动窗口片段
            for i in range(0, waveform.shape[1] - self.window_size, self.shift_size):
                segment = waveform[:, i:i+self.window_size]
                segments.append((segment, i))
        
        return segments
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        segment, position = self.segments[idx]
        return {
            'waveform': segment,
            'position': position
        }


class MultiSpeakerDataset(Dataset):
    def __init__(self, root_dir: str, target_speaker_id: str, other_speaker_ids: List[str], 
                 transform=None, window_size=100, shift_size=50, sample_rate=16000):
        """
        多说话人数据集，用于训练VSMask
        
        Args:
            root_dir: 数据集根目录
            target_speaker_id: 目标说话人ID（要保护的）
            other_speaker_ids: 其他说话人ID列表（用于误导）
            transform: 数据转换函数
            window_size: 滑动窗口大小
            shift_size: 滑动窗口移动步长
            sample_rate: 音频采样率
        """
        self.root_dir = root_dir
        self.target_dataset = VCTKDataset(root_dir, target_speaker_id, transform, 
                                         'train', window_size, shift_size, sample_rate)
        
        # 随机选择一个其他说话人作为目标误导说话人
        self.other_speaker_id = random.choice(other_speaker_ids)
        self.other_dataset = VCTKDataset(root_dir, self.other_speaker_id, transform, 
                                        'train', window_size, shift_size, sample_rate)
        
    def __len__(self) -> int:
        return len(self.target_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_data = self.target_dataset[idx]
        
        # 随机选择一个其他说话人的样本
        other_idx = random.randint(0, len(self.other_dataset) - 1)
        other_data = self.other_dataset[other_idx]
        
        return {
            'source_waveform': target_data['waveform'],
            'source_position': target_data['position'],
            'target_waveform': other_data['waveform'],
            'target_position': other_data['position'],
            'target_speaker_id': self.other_speaker_id
        }


def get_dataloaders(root_dir: str, target_speaker_id: str, other_speaker_ids: List[str],
                   batch_size: int = 32, window_size: int = 100, shift_size: int = 50,
                   sample_rate: int = 16000, num_workers: int = 4):
    """
    创建训练和测试数据加载器
    
    Args:
        root_dir: 数据集根目录
        target_speaker_id: 目标说话人ID
        other_speaker_ids: 其他说话人ID列表
        batch_size: 批处理大小
        window_size: 滑动窗口大小
        shift_size: 滑动窗口移动步长
        sample_rate: 音频采样率
        num_workers: 数据加载工作线程数
        
    Returns:
        train_loader, test_loader
    """
    # 创建训练集数据加载器
    train_dataset = MultiSpeakerDataset(
        root_dir, target_speaker_id, other_speaker_ids,
        window_size=window_size, shift_size=shift_size, sample_rate=sample_rate
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    # 创建测试集数据加载器
    test_dataset = VCTKDataset(
        root_dir, target_speaker_id, split='test',
        window_size=window_size, shift_size=shift_size, sample_rate=sample_rate
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader
