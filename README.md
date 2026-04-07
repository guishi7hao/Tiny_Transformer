# Tiny_Transformer
从零实现的轻量级 Transformer 模型，支持 Encoder-Decoder 架构、多头注意力、交叉注意力等核心组件。学习和理解 Transformer 内部机制。

✨ 特性
- 🔧 完整实现：从零实现 MultiHeadAttention、PositionalEncoding、LayerNorm、MLP 等核心模块

- 🎯 双模式注意力：统一接口支持自注意力（因果掩码）和交叉注意力（编解码器交互）

- ⚡ 性能优化：自动检测并使用 PyTorch 2.0+ 的 Flash Attention，回退到手动实现

- 🎲 生成策略：支持温度采样和 Top-K 采样，可用于文本生成任务

- 📊 配置管理：灵活的 Config 配置系统，便于调整模型超参数

- 🚀 优化器配置：自动选择fused AdamW优化器（CUDA环境）

🏗️ 架构概览  
输入序列  
- 输入序列 → Token Embedding → Positional Encoding
- Encoder (×N layers)
  - Multi-Head Self-Attention
  - Feed-Forward Network
  - Residual Connections + LayerNorm
- Decoder (×N layers)
  - Masked Multi-Head Self-Attention
  - Cross-Attention (with Encoder output)
  - Feed-Forward Network
  - Residual Connections + LayerNorm
- LM Head → 输出 logits

🚀 快速开始
```python
from tiny_transformer import Transformer
from collections import namedtuple

# 配置模型
Config = namedtuple("Config", ["n_embd", "n_head", "n_layers", "dropout", "bias", "block_size", "vocab_size"])

config = Config(
    n_embd=512,      # 嵌入维度
    n_head=8,        # 注意力头数
    n_layers=6,      # 编码器/解码器层数
    dropout=0.1,     # Dropout 概率
    bias=True,       # 是否使用偏置
    block_size=1024, # 最大序列长度
    vocab_size=10000 # 词表大小
)

# 创建模型
model = Transformer(config)

# 前向传播
src_ids = torch.randint(0, 10000, (4, 128))  # (batch, seq_len)
logits, loss = model(src_ids)
print(f"Output shape: {logits.shape}")  # (4, 128, 10000)
```
## 文本生成
```python
# 自回归生成
start_tokens = torch.tensor([[1, 2, 3, 4]])  # (1, seq_len)
generated = model.generate(
    idx=start_tokens,
    max_new_tokens=50,
    temperature=0.8,    # 控制随机性（0.8 = 探索，1.0 = 平衡，1.2 = 随机）
    top_k=40           # 只从概率最高的 40 个 token 中采样
)
```
## 📖 核心实现
### 1. 多头注意力（支持自注意力和交叉注意力）
```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, encoder_output=None):
        if encoder_output is None:
            # 自注意力：Q、K、V 来自同一个输入
            q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        else:
            # 交叉注意力：Q 来自解码器，K、V 来自编码器
            q = self.qkv(x)[:, :, :self.n_embd]  # 取 Q 部分
            k = self.k_proj_cross(encoder_output)
            v = self.v_proj_cross(encoder_output)
        
        # 多头分割 + 注意力计算
        # ...
```
### 2. Flash Attention 自动切换
```python
if hasattr(F, "scaled_dot_product_attention"):
    # 使用 PyTorch 2.0+ 的高效实现
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
else:
    # 手动实现注意力（后向兼容）
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    out = att @ v
```
### 3. Pre-Norm 残差结构
```python
# 原始 Post-Norm: x = LayerNorm(x + Attention(x))
# 改进 Pre-Norm: x = x + Attention(LayerNorm(x))  ✅ 梯度更稳定

def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```
## 🔧 配置参数
| 参数 | 类型 | 说明 | 典型值 |
|---------|--------|-----------------|-----------------|
| n_embd       | int    | 嵌入维度        |512, 768|
| n_head       | int    | 注意力头数      |8, 12|
| n_layers     | int    |编码器/解码器层数|6, 12|
| block_size   | int    |最大序列长度     |512, 1024|
| vocab_size   | int    |词表大小         |5000-50000|
| dropout      | float  |Dropout 概率     |0.1|
| bias         | bool   |是否使用偏置     |True|
## 📊 模型规模
| 配置 | n_embd | n_head | n_layers | 参数量 |
|------|--------|--------|----------|--------|
| Tiny | 128 | 4 | 2 |3.68M |
| Small | 256 | 8 | 4 | 14.07M |
| Medium | 384 | 12 | 6 | 37.85M |

## 🧪 测试
```python
# 运行基础测试
python test.py

# 预期输出
number of parameters: 0.02M
idx torch.Size([4, 8])
tok_emb torch.Size([4, 8, 16])
enc_out: torch.Size([4, 8, 16])
logits torch.Size([4, 8, 10])  # (batch, seq_len, vocab_size)
```
## 性能测试结果

测试环境：NVIDIA RTX2060, PyTorch 2.1.0, batch_size=4

| 模型规模 | 参数量 | Time(ms)| 显存占用 |
|---------|--------|-----------------|----------|
| Tiny    | 3.68M  | 3.75            | 30.37 MB  |
| Small   | 14.07M  |  9.07           | 77.97 MB   |
| Medium  | 37.85M  | 20.92           |  178.60 MB  |

## 📝 待办事项
- 添加 BPE 分词器
- 支持多 GPU 训练
- 实现学习率预热（Warmup）
- 添加模型保存/加载示例
- 在机器翻译任务上验证
## 🙏 致谢
- Attention Is All You Need - 原始 Transformer 论文
- Andrej Karpathy's nanoGPT - 实现参考
- The Annotated Transformer - 详细注解
