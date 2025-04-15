import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import get_dataloaders
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter
import tqdm

def train_universal_header(speaker_encoder, args):
    """
    训练通用扰动头
    
    Args:
        speaker_encoder: 目标语音合成模型的说话人编码器
        args: 命令行参数
    """
    print("开始训练通用扰动头...")
    
    # 创建数据加载器
    train_loader, _ = get_dataloaders(
        args.data_dir, args.target_speaker, args.other_speakers,
        batch_size=args.batch_size, window_size=args.window_size,
        shift_size=args.shift_size, sample_rate=args.sample_rate
    )
    
    # 初始化音频转换器
    converter = MelSpectrogramConverter(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # 初始化通用扰动头
    header = UniversalPerturbationHeader(
        mel_bins=args.n_mels,
        time_length=args.header_length,
        device=args.device
    )
    
    # 设置优化器
    optimizer = optim.Adam([header.header], lr=args.lr)
    
    # 收集训练样本
    source_mels = []
    target_mels = []
    
    print("收集训练样本...")
    for batch in tqdm.tqdm(train_loader):
        source_waveform = batch['source_waveform'].to(args.device)
        target_waveform = batch['target_waveform'].to(args.device)
        
        # 转换为梅尔频谱图
        for i in range(source_waveform.size(0)):
            source_mel = converter.waveform_to_mel(source_waveform[i]).unsqueeze(0)
            target_mel = converter.waveform_to_mel(target_waveform[i]).unsqueeze(0)
            
            # 仅使用header_length长度的片段
            if source_mel.shape[-1] >= args.header_length and target_mel.shape[-1] >= args.header_length:
                source_mels.append(source_mel[:, :, :, :args.header_length])
                target_mels.append(target_mel[:, :, :, :args.header_length])
        
        # 收集足够的样本后停止
        if len(source_mels) >= args.max_samples:
            break
    
    # 将样本堆叠成批次
    source_mels = torch.cat(source_mels[:args.max_samples], dim=0)
    target_mels = torch.cat(target_mels[:args.max_samples], dim=0)
    
    print(f"开始优化，收集了 {source_mels.size(0)} 个样本...")
    # 执行优化
    header.optimize(
        source_mels, target_mels, speaker_encoder,
        optimizer, num_iterations=args.iterations,
        epsilon=args.epsilon, lambda_param=args.lambda_param
    )
    
    # 保存通用扰动头
    os.makedirs(args.output_dir, exist_ok=True)
    header.save(os.path.join(args.output_dir, 'universal_header.pt'))
    print(f"通用扰动头已保存到 {os.path.join(args.output_dir, 'universal_header.pt')}")

def main():
    parser = argparse.ArgumentParser(description='训练VSMask通用扰动头')
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='./data/VCTK-Corpus',
                        help='数据集根目录')
    parser.add_argument('--target_speaker', type=str, required=True,
                        help='目标说话人ID（要保护的）')
    parser.add_argument('--other_speakers', type=str, nargs='+', required=True,
                        help='其他说话人ID列表（用于误导）')
    
    # 音频参数
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='音频采样率')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT大小')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='帧移')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='梅尔滤波器组数量')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--window_size', type=int, default=100,
                        help='滑动窗口大小')
    parser.add_argument('--shift_size', type=int, default=50,
                        help='滑动窗口移动步长')
    parser.add_argument('--header_length', type=int, default=100,
                        help='通用扰动头的时间长度')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='优化迭代次数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='扰动最大幅度')
    parser.add_argument('--lambda_param', type=float, default=0.5,
                        help='损失函数中的平衡参数')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='用于训练的最大样本数')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # TODO: 这里应该加载目标语音合成模型的说话人编码器
    # 这里只是一个示例，实际应用中需要加载真实的编码器
    class DummySpeakerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 128)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 加载或创建说话人编码器
    speaker_encoder = DummySpeakerEncoder().to(args.device)
    
    # 训练通用扰动头
    train_universal_header(speaker_encoder, args)

if __name__ == '__main__':
    main()
