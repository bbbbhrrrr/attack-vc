import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

from data.dataset import get_dataloaders
from models.predictive_model import PredictiveModel
from models.header_model import UniversalPerturbationHeader
from utils.audio import MelSpectrogramConverter

def train_predictive_model(speaker_encoder, args):
    """
    训练预测模型
    
    Args:
        speaker_encoder: 目标语音合成模型的说话人编码器
        args: 命令行参数
    """
    print("开始训练预测模型...")
    
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
    
    # 初始化预测模型
    model = PredictiveModel(
        mel_bins=args.n_mels,
        time_dim=args.window_size
    ).to(args.device)
    
    # 加载通用扰动头（如果存在）
    header = None
    if args.header_path and os.path.exists(args.header_path):
        header = UniversalPerturbationHeader(
            mel_bins=args.n_mels,
            time_length=args.header_length,
            device=args.device
        )
        header.load(args.header_path)
        print(f"已加载通用扰动头：{args.header_path}")
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        for batch in tqdm.tqdm(train_loader):
            source_waveform = batch['source_waveform'].to(args.device)
            target_waveform = batch['target_waveform'].to(args.device)
            
            # 转换为梅尔频谱图
            source_mels = []
            target_mels = []
            
            for i in range(source_waveform.size(0)):
                source_mel = converter.waveform_to_mel(source_waveform[i]).unsqueeze(0)
                target_mel = converter.waveform_to_mel(target_waveform[i]).unsqueeze(0)
                
                # 如果有通用扰动头，应用到输入上
                if header:
                    # 应用到前部分
                    header_length = min(source_mel.shape[-1], header.header.shape[-1])
                    source_mel[:, :, :, :header_length] += header.header[:, :, :, :header_length]
                
                source_mels.append(source_mel)
                target_mels.append(target_mel)
            
            source_mels = torch.cat(source_mels, dim=0)
            target_mels = torch.cat(target_mels, dim=0)
            
            # 预测扰动
            predicted_perturbation = model(source_mels)
            
            # 应用预测扰动
            future_idx = args.future_steps
            if future_idx < source_mels.shape[-1]:
                # 将预测扰动应用到未来时间步
                perturbed_mels = source_mels.clone()
                future_end = min(future_idx + predicted_perturbation.shape[-1], perturbed_mels.shape[-1])
                perturbed_mels[:, :, :, future_idx:future_end] += predicted_perturbation[:, :, :, :future_end-future_idx]
                
                # 应用基于权重的约束
                weighted_perturbed = converter.apply_weighted_constraint(
                    perturbed_mels - source_mels,
                    epsilon1=args.epsilon1,
                    epsilon2=args.epsilon2,
                    epsilon3=args.epsilon3
                )
                perturbed_mels = source_mels + weighted_perturbed
                
                # 获取说话人嵌入
                source_embedding = speaker_encoder(source_mels)
                target_embedding = speaker_encoder(target_mels)
                perturbed_embedding = speaker_encoder(perturbed_mels)
                
                # 计算损失：最小化与目标说话人的距离，最大化与源说话人的距离
                loss_target = nn.MSELoss()(perturbed_embedding, target_embedding)
                loss_source = nn.MSELoss()(perturbed_embedding, source_embedding)
                
                loss = loss_target - args.lambda_param * loss_source
                
                # 梯度更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
        # 计算平均损失并更新学习率
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f'predictive_model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
    
    # 保存最终模型
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'predictive_model_final.pt')
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到 {model_path}")

def main():
    parser = argparse.ArgumentParser(description='训练VSMask预测模型')
    
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
    parser.add_argument('--future_steps', type=int, default=10,
                        help='预测未来的时间步数')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epsilon1', type=float, default=0.1,
                        help='低频扰动最大幅度')
    parser.add_argument('--epsilon2', type=float, default=0.05,
                        help='中频扰动最大幅度')
    parser.add_argument('--epsilon3', type=float, default=0.08,
                        help='高频扰动最大幅度')
    parser.add_argument('--lambda_param', type=float, default=0.5,
                        help='损失函数中的平衡参数')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存模型的轮数间隔')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--header_path', type=str, default=None,
                        help='通用扰动头路径（如果存在）')
    
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
    
    # 训练预测模型
    train_predictive_model(speaker_encoder, args)

if __name__ == '__main__':
    main()
