import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from data_processor import LibriSpeechDataset, AudioMelDataset
from predictive_model import build_predictive_model

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args):
    # 检查数据路径是否存在
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        print("Creating directory structure...")
        os.makedirs(args.data_path, exist_ok=True)
        
    # 检查输出目录
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据集
    if args.dataset_type == "librispeech":
        train_dataset = LibriSpeechDataset(
            root=args.data_path,
            split="train-clean-100",
            n_mels=args.n_mels,
            input_length=args.input_length,
            output_length=args.output_length
        )
        val_dataset = LibriSpeechDataset(
            root=args.data_path,
            split="dev-clean",
            n_mels=args.n_mels,
            input_length=args.input_length,
            output_length=args.output_length
        )
    else:
        # 使用自定义音频数据集
        train_dataset = AudioMelDataset(
            data_path=args.data_path,
            split="train",
            n_mels=args.n_mels,
            input_length=args.input_length,
            output_length=args.output_length
        )
        val_dataset = AudioMelDataset(
            data_path=args.data_path,
            split="val",
            n_mels=args.n_mels,
            input_length=args.input_length,
            output_length=args.output_length
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 构建模型
    model = build_predictive_model(
        n_mels=args.n_mels,
        input_length=args.input_length,
        output_length=args.output_length
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for i, (input_mel, target_mel) in enumerate(train_progress):
            input_mel = input_mel.to(device)
            target_mel = target_mel.to(device)
            
            # 前向传播
            output = model(input_mel)
            loss = criterion(output, target_mel)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新训练损失
            train_loss += loss.item()
            train_progress.set_postfix({"loss": loss.item()})
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for i, (input_mel, target_mel) in enumerate(val_progress):
                input_mel = input_mel.to(device)
                target_mel = target_mel.to(device)
                
                # 前向传播
                output = model(input_mel)
                loss = criterion(output, target_mel)
                
                # 更新验证损失
                val_loss += loss.item()
                val_progress.set_postfix({"loss": loss.item()})
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, "best_model.pth")
            model.save_model(model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
        
        # 保存最新模型
        model_path = os.path.join(args.output_dir, "latest_model.pth")
        model.save_model(model_path)
        
        # 保存损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
        plt.close()
    
    print("Training completed!")

if __name__ == "__main__":
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Train the predictive model for VSMask")
    parser.add_argument("--data_path", type=str, 
                        default=os.path.join(script_dir, "..", "data", "LibriSpeech"),
                        help="Path to the dataset")
    parser.add_argument("--dataset_type", type=str, default="librispeech", 
                        choices=["librispeech", "custom"], 
                        help="Type of dataset to use")
    parser.add_argument("--output_dir", type=str, 
                        default=os.path.join(script_dir, "..", "models"),
                        help="Directory to save models and results")
    parser.add_argument("--n_mels", type=int, default=512, 
                        help="Number of mel frequency bins")
    parser.add_argument("--input_length", type=int, default=100, 
                        help="Input sequence length")
    parser.add_argument("--output_length", type=int, default=32, 
                        help="Output sequence length")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    parser.add_argument("--no_cuda", action="store_true", 
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # 转换相对路径为绝对路径
    args.data_path = os.path.abspath(args.data_path)
    args.output_dir = os.path.abspath(args.output_dir)
    
    print(f"Using data path: {args.data_path}")
    print(f"Using output directory: {args.output_dir}")
    
    train(args)
