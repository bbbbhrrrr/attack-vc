import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange


def e2e_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters,
) -> Tensor:
    """端到端攻击
    
    通过直接操作声音转换模型的输出结果来生成对抗样本。
    
    Args:
        model: 声音转换模型
        vc_src: 源语音特征（提供语言内容）
        vc_tgt: 目标语音特征（提供声音特征，需要被保护）
        adv_tgt: 攻击目标语音特征
        eps: 扰动幅度上限
        n_iters: 优化迭代次数
        
    Returns:
        添加扰动后的目标语音特征
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 初始化随机扰动
    opt = torch.optim.Adam([ptb])  # 优化器
    criterion = nn.MSELoss()  # 损失函数
    pbar = trange(n_iters)  # 进度条

    with torch.no_grad():
        org_out = model.inference(vc_src, vc_tgt)  # 原始转换结果
        tgt_out = model.inference(vc_src, adv_tgt)  # 目标转换结果

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # 使用tanh限制扰动范围在[-eps, eps]
        adv_out = model.inference(vc_src, adv_inp)  # 使用添加扰动后的输入进行推理
        # 损失函数：使输出接近攻击目标，同时远离原始输出
        loss = criterion(adv_out, tgt_out) - 0.1 * criterion(adv_out, org_out)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 返回添加扰动后的输入


def emb_attack(
    model: nn.Module, vc_tgt: Tensor, adv_tgt: Tensor, eps: float, n_iters: int
) -> Tensor:
    """嵌入攻击
    
    通过操作说话人编码器的输出嵌入向量来生成对抗样本。
    
    Args:
        model: 声音转换模型
        vc_tgt: 目标语音特征（提供声音特征，需要被保护）
        adv_tgt: 攻击目标语音特征
        eps: 扰动幅度上限
        n_iters: 优化迭代次数
        
    Returns:
        添加扰动后的目标语音特征
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 初始化随机扰动
    opt = torch.optim.Adam([ptb])  # 优化器
    criterion = nn.MSELoss()  # 损失函数
    pbar = trange(n_iters)  # 进度条

    with torch.no_grad():
        org_emb = model.speaker_encoder(vc_tgt)  # 原始说话人嵌入
        tgt_emb = model.speaker_encoder(adv_tgt)  # 目标说话人嵌入

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # 使用tanh限制扰动范围在[-eps, eps]
        adv_emb = model.speaker_encoder(adv_inp)  # 使用添加扰动后的输入获取说话人嵌入
        # 损失函数：使说话人嵌入接近攻击目标，同时远离原始嵌入
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 返回添加扰动后的输入


def fb_attack(
    model: nn.Module,
    vc_src: Tensor,
    vc_tgt: Tensor,
    adv_tgt: Tensor,
    eps: float,
    n_iters: int,
) -> Tensor:
    """反馈攻击
    
    通过操作转换后语音的说话人嵌入来生成对抗样本。
    
    Args:
        model: 声音转换模型
        vc_src: 源语音特征（提供语言内容）
        vc_tgt: 目标语音特征（提供声音特征，需要被保护）
        adv_tgt: 攻击目标语音特征
        eps: 扰动幅度上限
        n_iters: 优化迭代次数
        
    Returns:
        添加扰动后的目标语音特征
    """
    ptb = torch.zeros_like(vc_tgt).normal_(0, 1).requires_grad_(True)  # 初始化随机扰动
    opt = torch.optim.Adam([ptb])  # 优化器
    criterion = nn.MSELoss()  # 损失函数
    pbar = trange(n_iters)  # 进度条

    with torch.no_grad():
        org_emb = model.speaker_encoder(model.inference(vc_src, vc_tgt))  # 原始转换结果的说话人嵌入
        tgt_emb = model.speaker_encoder(adv_tgt)  # 目标说话人嵌入

    for _ in pbar:
        adv_inp = vc_tgt + eps * ptb.tanh()  # 使用tanh限制扰动范围在[-eps, eps]
        adv_emb = model.speaker_encoder(model.inference(vc_src, adv_inp))  # 使用添加扰动后的输入进行推理，然后获取说话人嵌入
        # 损失函数：使转换结果的说话人嵌入接近攻击目标，同时远离原始嵌入
        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return vc_tgt + eps * ptb.tanh()  # 返回添加扰动后的输入
