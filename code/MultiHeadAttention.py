import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple
from typing import Optional
import inspect


# 定义配置结构
Config = namedtuple("Config", ["n_embd", "n_head", "dropout", "bias", "block_size"])

"""多头注意力计算模块"""


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self, config, is_causal=False):
        super(MultiHeadAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # 自注意力：使用合并的QKV投影（一次性计算Q、K、V）
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # 交叉注意力：需要独立的K、V投影（因为K、V来自编码器，Q来自解码器）
        # 注意：交叉注意力的Q仍然使用qkv的前1/3部分

        self.k_proj_cross = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj_cross = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 输出投影矩阵返回残差流
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal

        # 检查是否支持Flash Attention (PyTorch 2.0+)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash and is_causal:
            self.register_buffer(
                "bias", torch.tril(torch.ones(config.block.size, config.blocksize))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(
            self,
            x:torch.Tensor,
            encoder_output:Optional[torch.Tensor]=None,
            ):

        B, T, C = x.size()

        if encoder_output is None:
            # ========== 自注意力模式 ==========
            # 一次性计算Q、K、V
            qkv = self.qkv(x).split(self.n_embd, dim=2)
            q, k, v = [
                y.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for y in qkv
            ]
            seq_len_k = T
        else:
            # ========== 交叉注意力模式 ==========
            # Q来自解码器输入（使用qkv的前1/3部分）
            qkv_partial= self.qkv(x)[:,:,:self.n_embd] # 只取前n_embd作为Q
            q = qkv_partial.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

            # K、V来自编码器输出
            B_enc, T_enc, C_enc = encoder_output.size()
            assert B == B_enc, f"Batch size mismatch: {B} vs {B_enc}"
            assert C == C_enc, f"Embedding dim mismatch: {C} vs {C_enc}"

            # 交叉注意力：需要独立的K、V投影（因为K、V来自编码器，Q来自解码器）
            k = self.k_proj_cross(encoder_output)
            v = self.v_proj_cross(encoder_output)

            seq_len_k = T_enc
            use_causal = False

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.is_causal,
            )
        else:
            # 手动实现注意力
            # 计算注意力分数: [B, H, T, seq_len_k]
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # 应用因果掩码（仅自注意力）
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            out = att @ v

        # 步骤4: 合并多头输出
        # 转置回: (B, T, n_head, head_dim)
        # 然后重塑为: (B, T, C)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.c_proj(out))

        return out


"""全连接模块"""


class MLP(nn.Module):
    """docstring for MLP.nn.Module( )"""

    def __init__(self, config):
        super().__init__()
        # Transformer 的全连接模块有两个线性层，中间加了一个 RELU 激活函数
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


"""层规范化模块"""


class LayerNorm(nn.Module):
    """docstring for LayerNorm."""

    def __init__(self, ndim, bias):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


"""Encoder Layer"""


class EncoderLayer(nn.Module):
    """docstring for EncoderLayer."""

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config, is_causal=False)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = x + self.mlp(self.ln_2(x))
        return x


"""Encoder"""


class Encoder(nn.Module):
    """docstring for Encoder."""

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = LayerNorm(config.n_embd, config.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


"""Decoder Layer"""


class DecoderLayer(nn.Module):
    """docstring for DecoderLayer."""

    def __init__(self, config):
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        super(DecoderLayer, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        #因果掩码自注意力
        self.Mask_attn = MultiHeadAttention(config, is_causal=True)

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        #交叉注意力
        self.attn = MultiHeadAttention(config, is_causal=False)

        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
            self,
            x:torch.Tensor,
            encoder_output:Optional[torch.Tensor]=None,
            ):
        # 1. 掩码自注意力
        x = self.ln_1(x)
        x = x + self.Mask_attn(x)

        # 2. 交叉注意力
        x = self.ln_2(x)
        x = x + self.attn(x,encoder_output = encoder_output)

        # 3. FFN
        x = x + self.mlp(self.ln_3(x))

        return x

class PositionalEncoding(nn.Module):
    # 在输入上加入了位置编码

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p=config.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(config.block_size, config.n_embd)
        position = torch.arange(0, config.block_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.n_embd, 2) * -(math.log(10000.0) / config.n_embd)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class Decoder(nn.Module):
    """docstring for Decoder."""
    def __init__(self, config):
        super(Decoder, self).__init__()
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.n_embd,bias = config.bias)
    
    def forward(
            self,
            x,
            encoder_output:Optional[torch.Tensor]=None,
            ):
        for layer in self.layers:
            x = layer(x,encoder_output = encoder_output)
        
        return self.norm(x)
    

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 必须输入词表大小和 block size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = PositionalEncoding(config),
            drop = nn.Dropout(config.dropout),
            encoder = Encoder(config),
            decoder = Decoder(config),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.config.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层得到的维度是 (batch size, sequence length, vocab_size, n_embd)，因此我们去掉倒数第二个维度
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    '''配置优化器'''
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # weight_decay: 权重衰减系数，learning_rate: 学习率，betas: AdamW 的 betas，device_type: 设备类型
        # 首先获取所有命名参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 参数根据维度分为两组。
        # 维度大于等于2的参数（通常是权重）会应用权重衰减，而维度小于2的参数（通常是偏置和层归一化参数）不会应用权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 打印一下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}； 总参数量为：{num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}, 总参数量为：{num_nodecay_params:,}")
        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。如果可用且 device_type 为 'cuda'，则使用融合版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"是否使用 fused AdamW: {use_fused}")

        return optimizer

    '''进行推理'''
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 推理阶段，输入为 idx，维度为 (batch size, sequence length)，max_new_tokens 为最大生成的 token 数量即按序推理 max_new_tokens 次
        for _ in range(max_new_tokens):
            # 如果输入序列太长，我们需要将它截断到 block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向计算，得到 logits，维度为 (batch size, sequence length, vocab size)
            logits, _ = self(idx_cond)
            # 使用最后一个 token 的 logits 作为当前输出，除以温度系数控制其多样性
            logits = logits[:, -1, :] / temperature
            # 如果使用 Top K 采样，将 logits 中除了 top_k 个元素的概率置为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 对输出结果进行 Softmax
            probs = F.softmax(logits, dim=-1)
            # 对结果概率进行采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将输出结果拼接到输入序列后面，作为下一次的输入
            idx = torch.cat((idx, idx_next), dim=1)
            # print("idx:", idx)

        return idx