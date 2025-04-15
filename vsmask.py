import os
import argparse
import torch
import numpy as np
import soundfile as sf
import torchaudio
import tqdm
from typing import Optional, Tuple

from models.predictive_model import PredictiveModel
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter

class VSMask:
    def __init__(self, predictive_model_path: str, header_path: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        VSMask系统类
        
        Args:
            predictive_model_path: 预测模型路径
            header_path: 通用扰动头路径
            device: 使用的设备
        """
        self.device = device
        
        # 加载预测模型
        self.predictive_model = PredictiveModel().to(device)
        self.predictive_model.load_state_dict(torch.load(predictive_model_path, map_location=device))
        self.predictive_model.eval()
        
        # 加载通用扰动头
        self.header = UniversalPerturbationHeader(device=device)
        self.header.load(header_path)
        
        # 初始化音频转换器
        self.converter = MelSpectrogramConverter()
        
        print(f"VSMask系统已初始化，使用设备: {device}")
    
    def protect_file(self, input_path: str, output_path: str, 
                     window_size: int = 100, future_step: int = 10,
                     epsilon1: float = 0.1, epsilon2: float = 0.05, epsilon3: float = 0.08) -> None:
        """
        保护音频文件
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径
            window_size: 滑动窗口大小
            future_step: 预测未来的时间步数
            epsilon1: 低频扰动最大幅度
            epsilon2: 中频扰动最大幅度
            epsilon3: 高频扰动最大幅度
        """
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(input_path)
        
        # 重采样到模型要求的采样率（如果需要）
        if sample_rate != self.converter.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.converter.sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.converter.sample_rate
        
        # 转换为单声道（如果是立体声）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 将波形移动到设备上
        waveform = waveform.to(self.device)
        
        # 应用保护
        protected_waveform = self._protect_waveform(
            waveform, window_size, future_step,
            epsilon1, epsilon2, epsilon3
        )
        
        # 保存保护后的音频
        torchaudio.save(output_path, protected_waveform.cpu(), sample_rate)
        print(f"保护后的音频已保存到 {output_path}")
    
    def protect_stream(self, input_stream, output_stream, 
                     window_size: int = 100, future_step: int = 10,
                     epsilon1: float = 0.1, epsilon2: float = 0.05, epsilon3: float = 0.08) -> None:
        """
        保护音频流（实时应用）
        
        Args:
            input_stream: 输入音频流
            output_stream: 输出音频流
            window_size: 滑动窗口大小
            future_step: 预测未来的时间步数
            epsilon1: 低频扰动最大幅度
            epsilon2: 中频扰动最大幅度
            epsilon3: 高频扰动最大幅度
        """
        # 这里只是一个示例，实际中需要根据具体的流处理逻辑进行修改
        buffer = []
        header_applied = False
        
        while True:
            # 读取音频块
            audio_chunk = input_stream.read(window_size)
            if not audio_chunk:
                break
            
            # 转换为torch张量
            chunk_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 如果还没应用header，先应用
            if not header_applied:
                # 转换为梅尔频谱图
                chunk_mel = self.converter.waveform_to_mel(chunk_tensor)
                
                # 应用header
                header_length = min(chunk_mel.shape[-1], self.header.header.shape[-1])
                chunk_mel[:, :, :, :header_length] += self.header.header[:, :, :, :header_length]
                
                # 转换回波形
                protected_chunk = self.converter.mel_to_waveform(chunk_mel)
                header_applied = True
            else:
                # 应用预测模型保护
                # 构建滑动窗口
                buffer.append(chunk_tensor)
                if len(buffer) > window_size // chunk_tensor.shape[1]:
                    buffer.pop(0)
                
                window = torch.cat(buffer, dim=1)
                
                # 转换为梅尔频谱图
                window_mel = self.converter.waveform_to_mel(window)
                
                # 预测扰动
                with torch.no_grad():
                    perturbation = self.predictive_model(window_mel)
                
                # 应用扰动到未来时间步
                future_mel = window_mel.clone()
                future_idx = future_step
                future_end = min(future_idx + perturbation.shape[-1], future_mel.shape[-1])
                future_mel[:, :, :, future_idx:future_end] += perturbation[:, :, :, :future_end-future_idx]
                
                # 应用基于权重的约束
                weighted_perturbed = self.converter.apply_weighted_constraint(
                    future_mel - window_mel,
                    epsilon1=epsilon1,
                    epsilon2=epsilon2,
                    epsilon3=epsilon3
                )
                future_mel = window_mel + weighted_perturbed
                
                # 转换回波形，取最后部分作为保护后的当前块
                future_wave = self.converter.mel_to_waveform(future_mel)
                protected_chunk = future_wave[:, -chunk_tensor.shape[1]:]
            
            # 写入输出流
            output_stream.write(protected_chunk.cpu().numpy())
    
    def _protect_waveform(self, waveform: torch.Tensor, window_size: int = 100, 
                         future_step: int = 10, epsilon1: float = 0.1, 
                         epsilon2: float = 0.05, epsilon3: float = 0.08) -> torch.Tensor:
        """
        保护波形数据
        
        Args:
            waveform: 输入波形 [1, T]
            window_size: 滑动窗口大小
            future_step: 预测未来的时间步数
            epsilon1: 低频扰动最大幅度
            epsilon2: 中频扰动最大幅度
            epsilon3: 高频扰动最大幅度
            
        Returns:
            保护后的波形 [1, T]
        """
        # 转换为梅尔频谱图
        mel_spec = self.converter.waveform_to_mel(waveform)
        
        # 应用通用扰动头到开始部分
        header_length = min(mel_spec.shape[-1], self.header.header.shape[-1])
        perturbed_mel = mel_spec.clone()
        perturbed_mel[:, :, :, :header_length] += self.header.header[:, :, :, :header_length]
        
        # 使用滑动窗口和预测模型生成实时扰动
        for start_idx in tqdm.tqdm(range(0, mel_spec.shape[-1] - window_size, future_step)):
            # 获取当前窗口
            window = mel_spec[:, :, :, start_idx:start_idx+window_size]
            
            # 预测扰动
            with torch.no_grad():
                perturbation = self.predictive_model(window)
            
            # 应用扰动到未来时间步
            future_idx = start_idx + window_size
            future_end = min(future_idx + perturbation.shape[-1], perturbed_mel.shape[-1])
            
            if future_idx < perturbed_mel.shape[-1]:
                perturbed_mel[:, :, :, future_idx:future_end] += perturbation[:, :, :, :future_end-future_idx]
        
        # 应用基于权重的约束
        weighted_perturbed = self.converter.apply_weighted_constraint(
            perturbed_mel - mel_spec,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
            epsilon3=epsilon3
        )
        perturbed_mel = mel_spec + weighted_perturbed
        
        # 转换回波形
        protected_waveform = self.converter.mel_to_waveform(perturbed_mel)
        
        return protected_waveform

def main():
    parser = argparse.ArgumentParser(description='VSMask: 语音合成防御系统')
    
    # 模型参数
    parser.add_argument('--predictive_model', type=str, required=True,
                        help='预测模型路径')
    parser.add_argument('--header', type=str, required=True,
                        help='通用扰动头路径')
    
    # 音频文件参数
    parser.add_argument('--input', type=str, required=True,
                        help='输入音频文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出音频文件路径')
    
    # 保护参数
    parser.add_argument('--window_size', type=int, default=100,
                        help='滑动窗口大小')
    parser.add_argument('--future_step', type=int, default=10,
                        help='预测未来的时间步数')
    parser.add_argument('--epsilon1', type=float, default=0.1,
                        help='低频扰动最大幅度')
    parser.add_argument('--epsilon2', type=float, default=0.05,
                        help='中频扰动最大幅度')
    parser.add_argument('--epsilon3', type=float, default=0.08,
                        help='高频扰动最大幅度')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备')
    
    args = parser.parse_args()
    
    # 初始化VSMask系统
    vsmask = VSMask(
        args.predictive_model,
        args.header,
        device=args.device
    )
    
    # 保护音频文件
    vsmask.protect_file(
        args.input,
        args.output,
        window_size=args.window_size,
        future_step=args.future_step,
        epsilon1=args.epsilon1,
        epsilon2=args.epsilon2,
        epsilon3=args.epsilon3
    )

if __name__ == '__main__':
    main()
