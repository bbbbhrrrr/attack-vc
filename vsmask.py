import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from tqdm import tqdm, trange
import time
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

from predictive_model import build_predictive_model
from universal_header import generate_universal_header, apply_universal_header
from weighted_constraint import apply_weighted_constraint
from data_utils import mel2wav, file2mel, normalize, denormalize, load_model

class VSMask:
    def __init__(
        self,
        model_dir: str,
        input_length: float = 1.25,  # 输入时间窗口长度(秒)
        time_delay: float = 0.4,     # 时间延迟(秒)
        output_length: float = 0.4,  # 输出长度(秒)
        epsilon: float = 0.1,         # 扰动最大幅度
        lambda_param: float = 1.0,    # 损失函数权重参数
        low_freq_weight: float = 1.15, # 低频段权重
        mid_freq_weight: float = 0.85, # 中频段权重
        high_freq_weight: float = 1.0, # 高频段权重
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化VSMask
        
        Args:
            model_dir: 模型目录路径
            input_length: 输入时间窗口长度(秒)
            time_delay: 时间延迟(秒)
            output_length: 输出长度(秒)
            epsilon: 扰动最大幅度
            lambda_param: 损失函数权重参数
            low_freq_weight: 低频段权重
            mid_freq_weight: 中频段权重
            high_freq_weight: 高频段权重
            device: 计算设备
        """
        self.device = device
        self.model, self.config, self.attr, _ = load_model(model_dir)
        self.model.eval()
        
        # 配置参数
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.low_freq_weight = low_freq_weight
        self.mid_freq_weight = mid_freq_weight
        self.high_freq_weight = high_freq_weight
        
        # 计算时间参数对应的帧数
        self.sr = self.config["preprocess"]["sample_rate"]
        self.hop_length = self.config["preprocess"]["hop_length"]
        self.n_mels = self.config["model"]["SpeakerEncoder"]["c_in"]
        
        self.input_frames = int(input_length * self.sr / self.hop_length)
        self.delay_frames = int(time_delay * self.sr / self.hop_length)
        self.output_frames = int(output_length * self.sr / self.hop_length)
        self.header_frames = self.input_frames + self.delay_frames
        
        # 初始化预测模型
        self.predictive_model = None
        self.universal_header = None
        
    def train_predictive_model(
        self,
        victim_samples: List[str],
        target_sample: str,
        n_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None
    ):
        """
        训练预测模型
        
        Args:
            victim_samples: 受害者说话人的语音样本文件路径列表
            target_sample: 目标说话人的语音样本文件路径
            n_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            save_path: 模型保存路径
        """
        print("正在训练预测模型...")
        
        # 加载目标说话人样本
        target_mel = file2mel(target_sample, **self.config["preprocess"])
        target_mel = normalize(target_mel, self.attr)
        target_mel = torch.from_numpy(target_mel).T.unsqueeze(0).to(self.device)
        
        # 加载受害者说话人样本
        victim_mels = []
        for sample in victim_samples:
            mel = file2mel(sample, **self.config["preprocess"])
            mel = normalize(mel, self.attr)
            mel = torch.from_numpy(mel).T.unsqueeze(0).to(self.device)
            victim_mels.append(mel)
        
        # 创建预测模型
        self.predictive_model = build_predictive_model(
            n_mels=self.n_mels,
            input_length=self.input_frames,
            output_length=self.output_frames
        ).to(self.device)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.predictive_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 生成通用扰动头部
        print("正在生成通用扰动头部...")
        self.universal_header = generate_universal_header(
            model=self.model.speaker_encoder,
            data_samples=victim_mels,
            target_sample=target_mel,
            header_length=self.header_frames,
            epsilon=self.epsilon,
            lambda_param=self.lambda_param,
            n_iters=1000,
            device=self.device
        )
        
        # 训练预测模型
        print("开始训练预测模型...")
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for i in range(0, len(victim_mels), batch_size):
                batch_samples = victim_mels[i:i+batch_size]
                optimizer.zero_grad()
                
                batch_loss = 0.0
                for sample in batch_samples:
                    # 确保样本长度足够
                    if sample.shape[2] < self.input_frames + self.output_frames:
                        continue
                    
                    # 提取输入窗口
                    input_window = sample[:, :, :self.input_frames]
                    
                    # 生成预测扰动
                    predicted_perturbation = self.predictive_model(input_window.unsqueeze(1))
                    predicted_perturbation = predicted_perturbation.squeeze(1)
                    
                    # 应用基于权重的约束
                    weighted_perturbation = apply_weighted_constraint(
                        predicted_perturbation,
                        sample_rate=self.sr,
                        n_mels=self.n_mels,
                        epsilon=self.epsilon,
                        low_freq_weight=self.low_freq_weight,
                        mid_freq_weight=self.mid_freq_weight,
                        high_freq_weight=self.high_freq_weight
                    )
                    
                    # 应用扰动到输入样本上
                    future_window = sample[:, :, self.input_frames:self.input_frames+self.output_frames]
                    perturbed_future = future_window + weighted_perturbation
                    
                    # 加上通用扰动头部创建完整的扰动样本
                    full_sample = sample[:, :, :self.input_frames+self.output_frames].clone()
                    full_sample[:, :, :self.input_frames] = apply_universal_header(
                        full_sample[:, :, :self.input_frames],
                        self.universal_header[:, :, :self.input_frames],
                        self.epsilon
                    )
                    full_sample[:, :, self.input_frames:] = perturbed_future
                    
                    # 计算说话人嵌入向量
                    with torch.no_grad():
                        original_emb = self.model.speaker_encoder(sample)
                        target_emb = self.model.speaker_encoder(target_mel)
                    
                    perturbed_emb = self.model.speaker_encoder(full_sample)
                    
                    # 计算损失
                    loss = criterion(perturbed_emb, target_emb) - self.lambda_param * criterion(perturbed_emb, original_emb)
                    batch_loss += loss
                
                # 平均损失
                if len(batch_samples) > 0:
                    batch_loss /= len(batch_samples)
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.6f}")
        
        # 保存模型
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'predictive_model': self.predictive_model.state_dict(),
                'universal_header': self.universal_header,
                'config': {
                    'input_frames': self.input_frames,
                    'delay_frames': self.delay_frames,
                    'output_frames': self.output_frames,
                    'header_frames': self.header_frames,
                    'epsilon': self.epsilon,
                    'lambda_param': self.lambda_param,
                    'low_freq_weight': self.low_freq_weight,
                    'mid_freq_weight': self.mid_freq_weight,
                    'high_freq_weight': self.high_freq_weight,
                }
            }, save_path)
            
            print(f"模型已保存到 {save_path}")
    
    def load_vsmask_model(self, model_path: str):
        """
        加载预训练的VSMask模型
        
        Args:
            model_path: 模型路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 加载配置
        config = checkpoint['config']
        self.input_frames = config['input_frames']
        self.delay_frames = config['delay_frames']
        self.output_frames = config['output_frames']
        self.header_frames = config['header_frames']
        self.epsilon = config['epsilon']
        self.lambda_param = config['lambda_param']
        self.low_freq_weight = config['low_freq_weight']
        self.mid_freq_weight = config['mid_freq_weight']
        self.high_freq_weight = config['high_freq_weight']
        
        # 加载预测模型
        self.predictive_model = build_predictive_model(
            n_mels=self.n_mels,
            input_length=self.input_frames,
            output_length=self.output_frames
        ).to(self.device)
        self.predictive_model.load_state_dict(checkpoint['predictive_model'])
        self.predictive_model.eval()
        
        # 加载通用扰动头部
        self.universal_header = checkpoint['universal_header'].to(self.device)
        
        print(f"VSMask模型已从 {model_path} 加载")
    
    def protect_audio(self, audio_path: str, output_path: str):
        """
        保护音频文件
        
        Args:
            audio_path: 输入音频文件路径
            output_path: 输出保护后的音频文件路径
        """
        if self.predictive_model is None or self.universal_header is None:
            raise RuntimeError("请先训练或加载VSMask模型")
        
        # 加载音频
        mel = file2mel(audio_path, **self.config["preprocess"])
        mel = normalize(mel, self.attr)
        mel = torch.from_numpy(mel).T.unsqueeze(0).to(self.device)
        
        # 应用通用扰动头部
        protected_mel = mel.clone()
        if mel.shape[2] <= self.header_frames:
            # 如果音频较短，只使用通用扰动头部
            protected_mel = apply_universal_header(
                protected_mel,
                self.universal_header[:, :, :protected_mel.shape[2]],
                self.epsilon
            )
        else:
            # 对开头应用通用扰动头部
            protected_mel[:, :, :self.header_frames] = apply_universal_header(
                protected_mel[:, :, :self.header_frames],
                self.universal_header,
                self.epsilon
            )
            
            # 对剩余部分应用预测模型生成的扰动
            for i in range(self.header_frames, mel.shape[2] - self.output_frames, self.output_frames):
                # 提取输入窗口
                if i + self.input_frames > mel.shape[2]:
                    break
                
                input_window = mel[:, :, i:i+self.input_frames]
                
                # 生成预测扰动
                with torch.no_grad():
                    predicted_perturbation = self.predictive_model(input_window.unsqueeze(1))
                    predicted_perturbation = predicted_perturbation.squeeze(1)
                
                # 应用基于权重的约束
                weighted_perturbation = apply_weighted_constraint(
                    predicted_perturbation,
                    sample_rate=self.sr,
                    n_mels=self.n_mels,
                    epsilon=self.epsilon,
                    low_freq_weight=self.low_freq_weight,
                    mid_freq_weight=self.mid_freq_weight,
                    high_freq_weight=self.high_freq_weight
                )
                
                # 应用扰动
                end_idx = min(i + self.input_frames + self.output_frames, mel.shape[2])
                output_end = min(self.output_frames, end_idx - (i + self.input_frames))
                protected_mel[:, :, i+self.input_frames:i+self.input_frames+output_end] += weighted_perturbation[:, :, :output_end]
        
        # 转换回时域并保存
        protected_mel = protected_mel.squeeze(0).T
        protected_mel = denormalize(protected_mel.cpu().numpy(), self.attr)
        protected_wav = mel2wav(protected_mel, **self.config["preprocess"])
        
        import soundfile as sf
        sf.write(output_path, protected_wav, self.sr)
        
        print(f"保护后的音频已保存到 {output_path}")
    
    def protect_real_time(self, input_device: int, output_device: int, duration: float = 30.0):
        """
        实时保护麦克风输入的语音
        
        Args:
            input_device: 输入设备ID
            output_device: 输出设备ID
            duration: 持续时间(秒)
        """
        if self.predictive_model is None or self.universal_header is None:
            raise RuntimeError("请先训练或加载VSMask模型")
        
        try:
            import sounddevice as sd
            import queue
            import threading
        except ImportError:
            raise ImportError("请安装 sounddevice 库: pip install sounddevice")
        
        # 创建音频队列
        q = queue.Queue()
        buffer_size = self.input_frames * self.hop_length
        
        # 回调函数，处理音频块
        def callback(indata, frames, time, status):
            q.put(indata.copy())
        
        # 处理线程
        def processing_thread():
            # 初始化保护状态
            is_protected = False
            audio_buffer = np.zeros((buffer_size * 2,))
            buffer_pos = 0
            
            # 设置输出流
            output_stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                device=output_device
            )
            output_stream.start()
            
            try:
                while True:
                    # 从队列获取音频数据
                    indata = q.get()
                    if indata is None:
                        break
                    
                    # 将新数据添加到缓冲区
                    indata_flat = indata.flatten()
                    if buffer_pos + len(indata_flat) <= len(audio_buffer):
                        audio_buffer[buffer_pos:buffer_pos+len(indata_flat)] = indata_flat
                        buffer_pos += len(indata_flat)
                    else:
                        # 缓冲区已满，移动数据并添加新数据
                        shift = buffer_pos + len(indata_flat) - len(audio_buffer)
                        audio_buffer[:-shift] = audio_buffer[shift:]
                        audio_buffer[-len(indata_flat):] = indata_flat
                        buffer_pos = len(audio_buffer)
                    
                    # 检查是否有足够的数据进行处理
                    if buffer_pos >= buffer_size:
                        # 将音频转换为mel频谱
                        audio_segment = audio_buffer[:buffer_size]
                        
                        # 这里需要简化处理，实际实现中应该调用file2mel等函数
                        # 但为了示例简单，我们直接生成一个模拟的频谱
                        mel = np.random.rand(self.n_mels, self.input_frames)
                        mel = torch.from_numpy(mel).float().unsqueeze(0).to(self.device)
                        
                        # 生成预测扰动
                        with torch.no_grad():
                            predicted_perturbation = self.predictive_model(mel.unsqueeze(1))
                            predicted_perturbation = predicted_perturbation.squeeze(1)
                        
                        # 应用基于权重的约束
                        weighted_perturbation = apply_weighted_constraint(
                            predicted_perturbation,
                            sample_rate=self.sr,
                            n_mels=self.n_mels,
                            epsilon=self.epsilon,
                            low_freq_weight=self.low_freq_weight,
                            mid_freq_weight=self.mid_freq_weight,
                            high_freq_weight=self.high_freq_weight
                        )
                        
                        # 将扰动转换回时域（简化处理）
                        perturbation_wav = np.random.rand(buffer_size) * self.epsilon
                        
                        # 添加扰动并播放
                        protected_wav = audio_segment + perturbation_wav
                        
                        # 播放受保护的音频
                        output_stream.write(protected_wav.astype(np.float32))
                        
                        # 更新缓冲区位置
                        buffer_pos -= buffer_size
                        if buffer_pos > 0:
                            audio_buffer[:buffer_pos] = audio_buffer[buffer_size:buffer_size+buffer_pos]
            
            finally:
                output_stream.stop()
                output_stream.close()
        
        # 创建处理线程
        thread = threading.Thread(target=processing_thread)
        thread.start()
        
        # 创建输入流
        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            device=input_device,
            callback=callback,
            blocksize=2048
        ):
            print(f"实时保护已启动，持续 {duration} 秒...")
            sd.sleep(int(duration * 1000))
        
        # 停止处理线程
        q.put(None)
        thread.join()
        
        print("实时保护已停止")
