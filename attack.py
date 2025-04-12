import argparse

import soundfile as sf
import torch

from attack_utils import e2e_attack, emb_attack, fb_attack
from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(
    model_dir: str,
    vc_src: str,
    vc_tgt: str,
    adv_tgt: str,
    output: str,
    eps: float,
    n_iters: int,
    attack_type: str,
):
    """执行对抗攻击
    
    对目标语音进行对抗攻击，使其在声音转换时表现异常。
    
    Args:
        model_dir: 模型文件目录
        vc_src: 源语音文件路径（提供语言内容，仅在e2e和fb攻击中使用）
        vc_tgt: 目标语音文件路径（需要保护的语音）
        adv_tgt: 攻击目标语音文件路径
        output: 输出文件路径
        eps: 扰动幅度上限
        n_iters: 优化迭代次数
        attack_type: 攻击类型 ('e2e', 'emb', 或 'fb')
    """
    # 检查参数有效性
    assert attack_type == "emb" or vc_src is not None
    
    # 加载模型和配置
    model, config, attr, device = load_model(model_dir)

    # 将音频文件转换为梅尔频谱图
    vc_tgt = file2mel(vc_tgt, **config["preprocess"])
    adv_tgt = file2mel(adv_tgt, **config["preprocess"])

    # 标准化
    vc_tgt = normalize(vc_tgt, attr)
    adv_tgt = normalize(adv_tgt, attr)

    # 转换为张量并移动到设备
    vc_tgt = torch.from_numpy(vc_tgt).T.unsqueeze(0).to(device)
    adv_tgt = torch.from_numpy(adv_tgt).T.unsqueeze(0).to(device)

    # 如果不是嵌入攻击，则需要源语音
    if attack_type != "emb":
        vc_src = file2mel(vc_src, **config["preprocess"])
        vc_src = normalize(vc_src, attr)
        vc_src = torch.from_numpy(vc_src).T.unsqueeze(0).to(device)

    # 根据攻击类型执行相应的攻击
    if attack_type == "e2e":  # 端到端攻击
        adv_inp = e2e_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "emb":  # 嵌入攻击
        adv_inp = emb_attack(model, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "fb":  # 反馈攻击
        adv_inp = fb_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    else:
        raise NotImplementedError()

    # 处理攻击结果
    adv_inp = adv_inp.squeeze(0).T
    adv_inp = denormalize(adv_inp.data.cpu().numpy(), attr)
    adv_inp = mel2wav(adv_inp, **config["preprocess"])

    # 保存结果
    sf.write(output, adv_inp, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="The directory of model files.")
    parser.add_argument(
        "vc_tgt",
        type=str,
        help="The target utterance to be defended, providing vocal timbre in voice conversion.",
    )
    parser.add_argument(
        "adv_tgt", type=str, help="The target used in adversarial attack."
    )
    parser.add_argument("output", type=str, help="The output defended utterance.")
    parser.add_argument(
        "--vc_src",
        type=str,
        default=None,
        help="The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="The maximum amplitude of the perturbation.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=1500,
        help="The number of iterations for updating the perturbation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["e2e", "emb", "fb"],
        default="emb",
        help="The type of adversarial attack to use (end-to-end, embedding, or feedback attack).",
    )
    main(**vars(parser.parse_args()))
