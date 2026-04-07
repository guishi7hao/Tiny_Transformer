import torch
import time
from collections import namedtuple
from tiny_transformer import Transformer

# 配置
Config = namedtuple("Config", ["n_embd", "n_head", "n_layer", "dropout", "bias", "block_size", "vocab_size"])

def quick_benchmark():
    """快速基准测试"""
    
    # 测试配置
    test_cases = [
        ("Tiny", 128, 4, 2, 512),
        ("Small", 256, 8, 4, 512),
        ("Medium", 384, 12, 6, 512),
    ]
    
    print(f"{'Model':<10} {'Params':<12} {'Time(ms)':<12} {'Memory(MB)':<12}")
    print("-" * 50)
    
    for name, n_embd, n_head, n_layers, seq_len in test_cases:
        # 创建配置
        config = Config(
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layers,
            dropout=0.0,
            bias=True,
            block_size=seq_len * 2,
            vocab_size=10000
        )
        
        # 创建模型
        model = Transformer(config).cuda()
        model.eval()
        
        # 计算参数量
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # 生成输入
        input_ids = torch.randint(0, 10000, (2, seq_len)).cuda()
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_ids)
        
        # 测时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        for _ in range(50):
            with torch.no_grad():
                _ = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        avg_time = (time.time() - start) / 50 * 1000
        
        # 测内存
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(input_ids)
            memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory = 0  # CPU 内存测量较复杂，暂不实现
        
        print(f"{name:<10} {n_params:<12.2f} {avg_time:<12.2f} {memory:<12.2f}")

if __name__ == "__main__":
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    quick_benchmark()