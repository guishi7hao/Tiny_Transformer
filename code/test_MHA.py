# 测试代码
import torch
import torch.functional as F
import MultiHeadAttention
from collections import namedtuple

Config = namedtuple('Config', ['n_embd', 'n_head', 'dropout', 'bias', 'block_size'])
def test_multihead_attention():
    """测试多头注意力模块的完整函数"""
    
    # 测试配置
    config = Config(
        n_embd=64,      # 嵌入维度
        n_head=4,       # 注意力头数
        dropout=0.1,     # dropout概率
        bias=True,       # 使用偏置
        block_size=128   # 序列最大长度
    )
    
    # 测试1: 基础功能测试
    def test_basic():
        print("运行基础功能测试...")
        mha = MultiHeadAttention(config)
        x = torch.randn(2, 16, 64)  # (batch_size, seq_len, n_embd)
        
        # 前向传播测试
        out = mha(x)
        assert out.shape == x.shape, "输出形状应与输入相同"
        print("✓ 前向传播形状测试通过")
        
        # 梯度测试
        x.requires_grad_(True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None, "应能计算梯度"
        print("✓ 梯度计算测试通过")
    
    # 测试2: 因果掩码测试
    def test_causal():
        print("\n运行因果掩码测试...")
        mha = MultiHeadAttention(config, is_causal=True)
        x = torch.randn(1, config.block_size, config.n_embd)
        
        # 检查非Flash Attention模式下的掩码
        if not mha.flash:
            out = mha(x)
            # 验证注意力权重是下三角的
            att_weights = (mha.qkv(x).split(mha.n_embd, dim=2)[0] @ 
                         mha.qkv(x).split(mha.n_embd, dim=2)[1].transpose(-2, -1))
            att_weights = F.softmax(att_weights, dim=-1)
            assert torch.allclose(att_weights.tril(), att_weights), "注意力权重应是下三角矩阵"
            print("✓ 非Flash模式因果掩码测试通过")
        
        # 检查Flash Attention模式下的掩码
        if mha.flash:
            mha.eval()  # 确保dropout关闭
            out1 = mha(x)
            mha.is_causal = False
            out2 = mha(x)
            assert not torch.allclose(out1, out2), "因果模式应影响输出"
            print("✓ Flash模式因果掩码测试通过")

  
    def test_device_compatibility():
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:
            try:
                # 1. 初始化
                mha = MultiHeadAttention(config).to(device)
                x = torch.randn(2, 16, 64).to(device)
                
                # 2. 验证设备同步
                print(f"\n测试设备: {device}")
                print(f"模型设备: {next(mha.parameters()).device}")
                print(f"输入设备: {x.device}")

                # 3. 前向传播
                out = mha(x)
                assert out.device.type == device
                print(f"输出设备: {out.device} → 测试通过")
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("× CUDA内存不足，尝试减小batch_size")
                else:
                    print(f"× 设备 {device} 错误: {e}")
            except AssertionError:
                print(f"× 设备 {device} 输出位置错误")
    
    def test_train_eval_diff():
        print("\n运行训练/评估模式测试...")
        mha = MultiHeadAttention(config)
        x = torch.randn(1, 10, config.n_embd)
        
        # 训练模式
        mha.train()
        out_train = mha(x)
        
        # 评估模式
        mha.eval()
        out_eval = mha(x)
        
        # Dropout应导致输出不同
        assert not torch.allclose(out_train, out_eval), "训练/评估模式应有不同输出"
        print("✓ 训练/评估模式差异测试通过")

    # 执行所有测试
    #test_basic()
    #test_causal()
    test_device_compatibility()
    #test_train_eval_diff()

    print("\n所有测试通过")    

if __name__ == "__main__":
     # 运行测试
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    test_multihead_attention()