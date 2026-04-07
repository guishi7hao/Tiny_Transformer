# 创建模型配置文件
# from dataclasses import dataclass
# from tiny_transformer import Transformer
# import torch

# @dataclass
# class TransformerConfig:
#     block_size: int = 1024
#     vocab_size: int = 50304 
#     n_layer: int = 4
#     n_head: int = 4
#     n_embd: int = 768
#     dropout: float = 0.0
#     bias: bool = True 

# model_config = TransformerConfig(
#     vocab_size=10,
#     block_size=12,
#     n_layer=2,
#     n_head=4, 
#     n_embd=16, 
#     dropout=0.0, 
#     bias=True)

# model = Transformer(model_config)
# idx = torch.randint(1, 10, (4, 8))
# print("idx",idx.size())
# logits, _ = model(idx)
# print("logits",logits.size())

# result = model.generate(idx, 3)
# print("generate result",result.size())
# print(result)
import torch
import torch.nn.functional as F

# 假设 vocab_size=5，只有5个词
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5, 0.3],  # 位置0的预测
    [0.5, 2.5, 0.2, 0.8, 0.1],  # 位置1的预测
    [0.1, 0.3, 3.0, 0.2, 0.4],  # 位置2的预测
])
targets = torch.tensor([1, 1, 2])
probs = F.softmax(logits, dim=-1)
print(probs)
correct_probs = probs[range(3), targets]
print(correct_probs)