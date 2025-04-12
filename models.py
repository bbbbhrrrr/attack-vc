from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm


def pad_layer(
    inp: Tensor, layer: nn.Module, pad_type: Optional[str] = "reflect"
) -> Tensor:
    """对输入进行填充并应用指定层
    
    Args:
        inp: 输入张量
        layer: 卷积层
        pad_type: 填充类型，默认为"reflect"
        
    Returns:
        填充并应用层后的输出
    """
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inp = F.pad(inp, pad=pad, mode=pad_type)
    out = layer(inp)
    return out


def pixel_shuffle_1d(inp: Tensor, scale_factor: Optional[float] = 2.0) -> Tensor:
    """一维像素重排（用于上采样）
    
    Args:
        inp: 输入张量
        scale_factor: 缩放因子，默认为2.0
        
    Returns:
        重排后的张量
    """
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def upsample(x: Tensor, scale_factor: Optional[float] = 2.0) -> Tensor:
    """使用最近邻插值进行上采样
    
    Args:
        x: 输入张量
        scale_factor: 缩放因子，默认为2.0
        
    Returns:
        上采样后的张量
    """
    x_up = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
    return x_up


def append_cond(x: Tensor, cond: Tensor) -> Tensor:
    """应用条件信息（均值和标准差）到输入
    
    Args:
        x: 输入张量
        cond: 条件张量（前半部分为均值，后半部分为标准差）
        
    Returns:
        应用条件后的张量
    """
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out


def conv_bank(
    x: Tensor,
    module_list: nn.Module,
    act: nn.Module,
    pad_type: Optional[str] = "reflect",
) -> Tensor:
    """应用卷积组（多种尺寸的卷积核）
    
    Args:
        x: 输入张量
        module_list: 卷积层列表
        act: 激活函数
        pad_type: 填充类型，默认为"reflect"
        
    Returns:
        卷积组输出结果（所有卷积结果拼接）
    """
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out


def get_act(act: str) -> nn.Module:
    """获取激活函数
    
    Args:
        act: 激活函数类型
        
    Returns:
        激活函数模块
    """
    if act == "lrelu":
        return nn.LeakyReLU()
    return nn.ReLU()


class ContentEncoder(nn.Module):
    """内容编码器
    
    提取输入语音的内容信息，不包含说话人特征。
    """
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        """初始化内容编码器
        
        Args:
            c_in: 输入通道数
            c_h: 隐藏层通道数
            c_out: 输出通道数
            kernel_size: 卷积核大小
            bank_size: 卷积组最大核大小
            bank_scale: 卷积组核大小增长步长
            c_bank: 卷积组通道数
            n_conv_blocks: 卷积块数量
            subsample: 每个卷积块的下采样率
            act: 激活函数类型
            dropout_rate: Dropout比例
        """
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_bank, kernel_size=k)
                for k in range(bank_scale, bank_size + 1, bank_scale)
            ]
        )
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            mu: 均值向量
            log_sigma: 对数标准差向量
        """
        out = conv_bank(x, self.conv_bank, act=self.act)
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return mu, log_sigma


class SpeakerEncoder(nn.Module):
    """说话人编码器
    
    提取输入语音的说话人特征。
    """
    def __init__(
        self,
        c_in: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        bank_size: int,
        bank_scale: int,
        c_bank: int,
        n_conv_blocks: int,
        n_dense_blocks: int,
        subsample: List[int],
        act: str,
        dropout_rate: float,
    ):
        """初始化说话人编码器
        
        Args:
            c_in: 输入通道数
            c_h: 隐藏层通道数
            c_out: 输出通道数
            kernel_size: 卷积核大小
            bank_size: 卷积组最大核大小
            bank_scale: 卷积组核大小增长步长
            c_bank: 卷积组通道数
            n_conv_blocks: 卷积块数量
            n_dense_blocks: 全连接块数量
            subsample: 每个卷积块的下采样率
            act: 激活函数类型
            dropout_rate: Dropout比例
        """
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_bank, kernel_size=k)
                for k in range(bank_scale, bank_size + 1, bank_scale)
            ]
        )
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp: Tensor) -> Tensor:
        """卷积块序列
        
        Args:
            inp: 输入特征
            
        Returns:
            处理后的特征
        """
        out = inp
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp: Tensor) -> Tensor:
        """全连接块序列
        
        Args:
            inp: 输入特征
            
        Returns:
            处理后的特征
        """
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x: Tensor) -> Tensor:
        """前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            说话人嵌入向量
        """
        out = conv_bank(x, self.conv_bank, act=self.act)
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        out = self.conv_blocks(out)
        out = self.pooling_layer(out).squeeze(-1)
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


class Decoder(nn.Module):
    """解码器
    
    根据内容特征和说话人嵌入生成目标语音特征。
    """
    def __init__(
        self,
        c_in: int,
        c_cond: int,
        c_h: int,
        c_out: int,
        kernel_size: int,
        n_conv_blocks: int,
        upsample: List[int],
        act: str,
        sn: bool,
        dropout_rate: float,
    ):
        """初始化解码器
        
        Args:
            c_in: 输入通道数
            c_cond: 条件通道数
            c_h: 隐藏层通道数
            c_out: 输出通道数
            kernel_size: 卷积核大小
            n_conv_blocks: 卷积块数量
            upsample: 每个卷积块的上采样率
            act: 激活函数类型
            sn: 是否使用谱归一化
            dropout_rate: Dropout比例
        """
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size))
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size))
                for _, up in zip(range(n_conv_blocks), self.upsample)
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
            [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks * 2)]
        )
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """前向传播
        
        Args:
            z: 内容特征
            cond: 条件信息（说话人嵌入）
            
        Returns:
            生成的语音特征
        """
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l * 2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l * 2 + 1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l])
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out


class AdaInVC(nn.Module):
    """AdaIN-VC 声音转换模型
    
    使用自适应实例归一化进行声音转换的模型。
    """
    def __init__(self, config: Dict):
        """初始化模型
        
        Args:
            config: 模型配置字典
        """
        super(AdaInVC, self).__init__()
        self.content_encoder = ContentEncoder(**config["ContentEncoder"])
        self.speaker_encoder = SpeakerEncoder(**config["SpeakerEncoder"])
        self.decoder = Decoder(**config["Decoder"])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """前向传播（训练模式）
        
        Args:
            x: 输入特征
            
        Returns:
            mu: 内容均值
            log_sigma: 内容对数标准差
            emb: 说话人嵌入
            dec: 解码结果
        """
        mu, log_sigma = self.content_encoder(x)
        emb = self.speaker_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        return mu, log_sigma, emb, dec

    def inference(self, src: Tensor, tgt: Tensor) -> Tensor:
        """推理（声音转换）
        
        Args:
            src: 源语音特征（提供语言内容）
            tgt: 目标语音特征（提供声音特征）
            
        Returns:
            转换后的语音特征
        """
        mu, _ = self.content_encoder(src)
        emb = self.speaker_encoder(tgt)
        dec = self.decoder(mu, emb)
        return dec
