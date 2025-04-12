import argparse

import soundfile as sf
import torch

from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(model_dir: str, source: str, target: str, output: str):
    """执行声音转换
    
    加载模型并进行声音转换，将结果保存到指定路径。
    
    Args:
        model_dir: 模型文件目录
        source: 源语音文件路径（提供语言内容）
        target: 目标语音文件路径（提供声音特征）
        output: 输出文件路径
    """
    # 加载模型和配置
    model, config, attr, device = load_model(model_dir)

    # 将音频文件转换为梅尔频谱图
    src_mel = file2mel(source, **config["preprocess"])
    tgt_mel = file2mel(target, **config["preprocess"])

    # 标准化
    src_mel = normalize(src_mel, attr)
    tgt_mel = normalize(tgt_mel, attr)

    # 转换为张量并移动到设备
    src_mel = torch.from_numpy(src_mel).T.unsqueeze(0).to(device)
    tgt_mel = torch.from_numpy(tgt_mel).T.unsqueeze(0).to(device)

    # 执行声音转换
    with torch.no_grad():
        out_mel = model.inference(src_mel, tgt_mel)
        out_mel = out_mel.squeeze(0).T
    
    # 反标准化
    out_mel = denormalize(out_mel.data.cpu().numpy(), attr)
    
    # 转换回波形
    out_wav = mel2wav(out_mel, **config["preprocess"])

    # 保存结果
    sf.write(output, out_wav, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="The directory of model files.")
    parser.add_argument(
        "source", type=str, help="The source utterance providing linguistic content."
    )
    parser.add_argument(
        "target", type=str, help="The target utterance providing vocal timbre."
    )
    parser.add_argument("output", type=str, help="The output converted utterance.")
    main(**vars(parser.parse_args()))
