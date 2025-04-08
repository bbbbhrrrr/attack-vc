import os
import argparse
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from vsmask import VSMask
from data_utils import load_model, file2mel, normalize, denormalize, mel2wav
from models import AdaInVC

def evaluate_asv_scores(
    model_dir: str,
    victim_dir: str,
    target_dir: str,
    output_dir: str,
    vsmask_model_path: Optional[str] = None,
    num_samples: int = 10,
    epsilon: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    评估VSMask对ASV系统的保护效果
    
    Args:
        model_dir: 语音合成模型目录
        victim_dir: 受害者说话人音频目录
        target_dir: 目标说话人音频目录
        output_dir: 输出目录
        vsmask_model_path: VSMask模型路径，如果为None则训练新模型
        num_samples: 评估样本数量
        epsilon: 扰动最大幅度
        device: 计算设备
    """
    try:
        import speechbrain as sb
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        raise ImportError("请安装 speechbrain 库: pip install speechbrain")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载语音合成模型
    vc_model, config, attr, _ = load_model(model_dir)
    vc_model.eval()
    
    # 加载ASV模型
    asv_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    # 获取受害者和目标说话人音频文件列表
    victim_files = [os.path.join(victim_dir, f) for f in os.listdir(victim_dir) if f.endswith('.wav')]
    target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.wav')]
    
    if len(victim_files) < num_samples or len(target_files) < 1:
        raise ValueError(f"受害者目录需要至少 {num_samples} 个音频文件，目标目录需要至少1个音频文件")
    
    # 随机选择评估样本
    np.random.seed(42)
    selected_victim_files = np.random.choice(victim_files, min(num_samples, len(victim_files)), replace=False)
    selected_target_file = np.random.choice(target_files, 1)[0]
    
    # 初始化VSMask
    vsmask = VSMask(
        model_dir=model_dir,
        epsilon=epsilon,
        device=device
    )
    
    # 加载或训练VSMask模型
    if vsmask_model_path is not None and os.path.exists(vsmask_model_path):
        vsmask.load_vsmask_model(vsmask_model_path)
    else:
        # 选择训练样本
        training_victim_files = np.random.choice(
            [f for f in victim_files if f not in selected_victim_files],
            min(10, len(victim_files) - len(selected_victim_files)),
            replace=False
        )
        
        # 训练VSMask模型
        vsmask.train_predictive_model(
            victim_samples=training_victim_files,
            target_sample=selected_target_file,
            n_epochs=5,
            batch_size=4,
            save_path=vsmask_model_path if vsmask_model_path else os.path.join(output_dir, "vsmask_model.pth")
        )
    
    # 评估结果
    results = {
        "raw_similarity": [],
        "protected_similarity": [],
        "synthetic_from_raw_similarity": [],
        "synthetic_from_protected_similarity": []
    }
    
    # 针对每个评估样本进行测试
    for i, victim_file in enumerate(tqdm(selected_victim_files)):
        # 文件名（不含路径和扩展名）
        file_name = os.path.splitext(os.path.basename(victim_file))[0]
        
        # 保护音频
        protected_file = os.path.join(output_dir, f"{file_name}_protected.wav")
        vsmask.protect_audio(victim_file, protected_file)
        
        # 生成合成语音（从原始音频）
        synthetic_from_raw_file = os.path.join(output_dir, f"{file_name}_synthetic_raw.wav")
        
        # 加载原始音频和目标音频
        src_mel = file2mel(selected_target_file, **config["preprocess"])
        tgt_mel = file2mel(victim_file, **config["preprocess"])
        
        src_mel = normalize(src_mel, attr)
        tgt_mel = normalize(tgt_mel, attr)
        
        src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(device)
        tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_mel = vc_model.inference(src_mel, tgt_mel)
            out_mel = out_mel.squeeze(0).T
        
        out_mel = denormalize(out_mel.cpu().numpy(), attr)
        out_wav = mel2wav(out_mel, **config["preprocess"])
        
        sf.write(synthetic_from_raw_file, out_wav, config["preprocess"]["sample_rate"])
        
        # 生成合成语音（从保护音频）
        synthetic_from_protected_file = os.path.join(output_dir, f"{file_name}_synthetic_protected.wav")
        
        # 加载保护音频
        protected_mel = file2mel(protected_file, **config["preprocess"])
        protected_mel = normalize(protected_mel, attr)
        protected_mel = torch.from_numpy(protected_mel).T.unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_mel = vc_model.inference(src_mel, protected_mel)
            out_mel = out_mel.squeeze(0).T
        
        out_mel = denormalize(out_mel.cpu().numpy(), attr)
        out_wav = mel2wav(out_mel, **config["preprocess"])
        
        sf.write(synthetic_from_protected_file, out_wav, config["preprocess"]["sample_rate"])
        
        # 计算ASV相似度分数
        # 原始音频与自身的相似度
        raw_emb = asv_model.encode_batch(torch.tensor([victim_file]))
        raw_similarity = torch.nn.functional.cosine_similarity(raw_emb, raw_emb).item()
        
        # 原始音频与保护音频的相似度
        protected_emb = asv_model.encode_batch(torch.tensor([protected_file]))
        protected_similarity = torch.nn.functional.cosine_similarity(raw_emb, protected_emb).item()
        
        # 原始音频与从原始音频合成的语音的相似度
        synthetic_raw_emb = asv_model.encode_batch(torch.tensor([synthetic_from_raw_file]))
        synthetic_from_raw_similarity = torch.nn.functional.cosine_similarity(raw_emb, synthetic_raw_emb).item()
        
        # 原始音频与从保护音频合成的语音的相似度
        synthetic_protected_emb = asv_model.encode_batch(torch.tensor([synthetic_from_protected_file]))
        synthetic_from_protected_similarity = torch.nn.functional.cosine_similarity(raw_emb, synthetic_protected_emb).item()
        
        # 记录结果
        results["raw_similarity"].append(raw_similarity)
        results["protected_similarity"].append(protected_similarity)
        results["synthetic_from_raw_similarity"].append(synthetic_from_raw_similarity)
        results["synthetic_from_protected_similarity"].append(synthetic_from_protected_similarity)
    
    # 计算平均结果
    avg_results = {k: np.mean(v) for k, v in results.items()}
    
    # 打印结果
    print("\n===== VSMask评估结果 =====")
    print(f"原始音频与自身的相似度: {avg_results['raw_similarity']:.4f}")
    print(f"原始音频与保护音频的相似度: {avg_results['protected_similarity']:.4f}")
    print(f"原始音频与从原始音频合成的语音的相似度: {avg_results['synthetic_from_raw_similarity']:.4f}")
    print(f"原始音频与从保护音频合成的语音的相似度: {avg_results['synthetic_from_protected_similarity']:.4f}")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    x = ['Raw', 'Protected', 'Synthetic\nfrom Raw', 'Synthetic\nfrom Protected']
    y = [
        avg_results['raw_similarity'],
        avg_results['protected_similarity'],
        avg_results['synthetic_from_raw_similarity'],
        avg_results['synthetic_from_protected_similarity']
    ]
    
    threshold = 0.25  # ASV阈值
    
    plt.bar(x, y, color=['blue', 'green', 'red', 'purple'])
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'ASV Threshold ({threshold})')
    plt.ylabel('Cosine Similarity Score')
    plt.title('VSMask Protection Effect on ASV System')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'vsmask_asv_results.png'))
    print(f"评估结果已保存到 {os.path.join(output_dir, 'vsmask_asv_results.png')}")
    
    return avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估VSMask对ASV系统的保护效果")
    parser.add_argument("--model_dir", type=str, required=True, help="语音合成模型目录")
    parser.add_argument("--victim_dir", type=str, required=True, help="受害者说话人音频目录")
    parser.add_argument("--target_dir", type=str, required=True, help="目标说话人音频目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--vsmask_model", type=str, default=None, help="VSMask模型路径，如不存在则训练新模型")
    parser.add_argument("--num_samples", type=int, default=10, help="评估样本数量")
    parser.add_argument("--epsilon", type=float, default=0.1, help="扰动最大幅度")
    
    args = parser.parse_args()
    
    evaluate_asv_scores(
        model_dir=args.model_dir,
        victim_dir=args.victim_dir,
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        vsmask_model_path=args.vsmask_model,
        num_samples=args.num_samples,
        epsilon=args.epsilon
    )
