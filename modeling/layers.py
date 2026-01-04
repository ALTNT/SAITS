"""
Layer modules for self-attention models (Transformer and SAITS).

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature#d_k**0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):#n_head = 8, d_model = 256, d_k = 32, d_v = 32, attn_dropout = 0.0
        super().__init__()

        self.n_head = n_head#8
        self.d_k = d_k#32
        self.d_v = d_v#32

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)#torch.Size([128, 48, 256])
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time,
        d_feature,#74
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        attn_dropout=0.1,
        **kwargs
    ):
        super(EncoderLayer, self).__init__()

        self.diagonal_attention_mask = kwargs["diagonal_attention_mask"]#True
        self.device = kwargs["device"]
        self.d_time = d_time#48
        self.d_feature = d_feature#74

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)#8,256,32,32,0
        self.dropout = nn.Dropout(dropout)#0.0
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)#256 512 0

    def forward(self, enc_input):
        if self.diagonal_attention_mask:#True
            mask_time = torch.eye(self.d_time).to(self.device)#torch.Size([48, 48])
        else:
            mask_time = None

        residual = enc_input#torch.Size([128, 48, 256])
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        # Post Norm的结构迁移性能更加好，也就是说在Pretraining中，Pre Norm和Post Norm都能做到大致相同的结果，
        # 但是Post Norm的Finetune效果明显更好。
        # 可能读者会反问《On Layer Normalization in the Transformer Architecture》不是显示Pre Norm要好于Post Norm吗？
        # 这是不是矛盾了？其实这篇文章比较的是在完全相同的训练设置下Pre Norm的效果要优于Post Norm，
        # 这只能显示出Pre Norm更容易训练，因为Post Norm要达到自己的最优效果，不能用跟Pre Norm一样的训
        # 练配置（比如Pre Norm可以不加Warmup但Post Norm通常要加），所以结论并不矛盾。
        enc_input = self.layer_norm(enc_input)#torch.Size([128, 48, 256])
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)#PositionWiseFeedForward
        return enc_output, attn_weights


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()
#         # Not a parameter
#         self.register_buffer(
#             "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
#         )

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         """Sinusoid position encoding table"""

#         def get_position_angle_vec(position):
#             return [
#                 position / np.power(10000, 2 * (hid_j // 2) / d_hid)
#                 for hid_j in range(d_hid)
#             ]

#         sinusoid_table = np.array(
#             [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
#         )
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)

#     def forward(self, x):
#         return x + self.pos_table[:, : x.size(1)].clone().detach()

# PositionalEncoding(out_dim, 366+2, T=1000)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, T: int = 10000):
        super(PositionalEncoding, self).__init__()
        # self.pe = nn.Parameter(torch.randn(max_len, d_model).unsqueeze(0))#位置编码
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)#torch.Size([32, 64])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#torch.Size([32, 1])
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(T)) / d_model))#torch.Size([32])div_term=10000 ^(− 2i/d)
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码矩阵注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x , doy= None):#x:torch.Size([4096, 30, 64])doy: torch.Size([4096, 30])
        # x 的形状: (batch_size, seq_len, d_model)
        # 将位置编码与输入相加
        if doy is not None:
            tmp = self.pe[0][doy, :]#torch.Size([4096, 75, 64])
            x = x + tmp
        else:
            tmp = self.pe[:, :x.size(1), :]#torch.Size([1, 30, 64])
            x = x + tmp#self.pe[:x.size(1), :] 从位置编码矩阵 self.pe 中截取前 seq_len 行，形状为 (seq_len, d_model)。
        # x = x + self.pe[:, :x.size(1)]

        return x