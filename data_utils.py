import copy
import os
import pickle
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.signal import lfilter

from models import AdaInVC


def inv_mel_matrix(sample_rate: int, n_fft: int, n_mels: int) -> np.array:
    """计算梅尔滤波器组的伪逆矩阵
    
    用于将梅尔频谱图转换回线性频谱图。
    
    Args:
        sample_rate: 采样率
        n_fft: FFT窗口大小
        n_mels: 梅尔滤波器组数量
        
    Returns:
        梅尔滤波器组的伪逆矩阵
    """
    m = librosa.filters.mel(sample_rate, n_fft, n_mels)
    p = np.matmul(m, m.T)
    d = [1.0 / x if np.abs(x) > 1e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m.T, np.diag(d))


def normalize(mel: np.array, attr: Dict) -> np.array:
    """标准化梅尔频谱图
    
    Args:
        mel: 梅尔频谱图
        attr: 属性字典，包含均值和标准差
        
    Returns:
        标准化后的梅尔频谱图
    """
    mean, std = attr["mean"], attr["std"]
    mel = (mel - mean) / std
    return mel


def denormalize(mel: np.array, attr: Dict) -> np.array:
    """反标准化梅尔频谱图
    
    Args:
        mel: 标准化的梅尔频谱图
        attr: 属性字典，包含均值和标准差
        
    Returns:
        原始梅尔频谱图
    """
    mean, std = attr["mean"], attr["std"]
    mel = mel * std + mean
    return mel


def file2mel(
    audio_path: str,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    """将音频文件转换为梅尔频谱图
    
    Args:
        audio_path: 音频文件路径
        sample_rate: 采样率
        preemph: 预加重系数
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        n_mels: 梅尔滤波器组数量
        ref_db: 参考分贝值
        max_db: 最大分贝值
        top_db: 静音修剪的阈值
        
    Returns:
        梅尔频谱图
    """
    # 加载音频
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    
    # 修剪静音
    wav, _ = librosa.effects.trim(wav, top_db=top_db)
    
    # 预加重
    wav = np.append(wav[0], wav[1:] - preemph * wav[:-1])
    
    # 计算线性频谱
    linear = librosa.stft(
        y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    mag = np.abs(linear)

    # 计算梅尔频谱
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels)
    mel = np.dot(mel_basis, mag)

    # 转换为分贝尺度并归一化
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)

    return mel


def mel2wav(
    mel: np.array,
    sample_rate: int,
    preemph: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    ref_db: float,
    max_db: float,
    top_db: float,
) -> np.array:
    """将梅尔频谱图转换为波形
    
    Args:
        mel: 梅尔频谱图
        sample_rate: 采样率
        preemph: 预加重系数
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        n_mels: 梅尔滤波器组数量
        ref_db: 参考分贝值
        max_db: 最大分贝值
        top_db: 静音修剪的阈值
        
    Returns:
        波形数据
    """
    # 转置并还原分贝尺度
    mel = mel.T
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    mel = np.power(10.0, mel * 0.05)
    
    # 转换回线性频谱
    inv_mat = inv_mel_matrix(sample_rate, n_fft, n_mels)
    mag = np.dot(inv_mat, mel)
    
    # Griffin-Lim算法重建相位
    wav = griffin_lim(mag, hop_length, win_length, n_fft)
    
    # 反预加重
    wav = lfilter([1], [1, -preemph], wav)

    return wav.astype(np.float32)


def griffin_lim(
    spect: np.array,
    hop_length: int,
    win_length: int,
    n_fft: int,
    n_iter: Optional[int] = 100,
) -> np.array:
    """Griffin-Lim算法重建相位
    
    根据幅度谱重建相位信息。
    
    Args:
        spect: 幅度谱
        hop_length: 帧移
        win_length: 窗口长度
        n_fft: FFT窗口大小
        n_iter: 迭代次数
        
    Returns:
        重建的波形
    """
    X_best = copy.deepcopy(spect)
    for _ in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spect * phase
    X_t = librosa.istft(X_best, hop_length, win_length, window="hann")
    y = np.real(X_t)
    return y


def load_model(model_dir: str) -> Tuple[nn.Module, Dict, Dict, str]:
    """加载模型和相关配置
    
    Args:
        model_dir: 模型文件目录
        
    Returns:
        model: 加载的模型
        config: 模型配置
        attr: 数据属性（均值和标准差）
        device: 计算设备
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_path = os.path.join(model_dir, "attr.pkl")
    cfg_path = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model.ckpt")

    # 加载数据属性、配置和模型
    attr = pickle.load(open(attr_path, "rb"))
    config = yaml.safe_load(open(cfg_path, "r"))
    model = AdaInVC(config["model"]).to(device)
    model.load_state_dict(torch.load(model_path))

    return model, config, attr, device
