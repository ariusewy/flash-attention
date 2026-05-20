import torch
import argparse
import time
import torch.cuda.profiler as profiler
import nvtx
# 这里的导入路径参考了你提供的代码片段
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
except ImportError:
    print("错误: 无法找到 Flash Attention 3 接口，请确保已正确安装 FA3 库。")
    exit(1)

def run_fmha():
    parser = argparse.ArgumentParser(description="Flash Attention 3 Single Run")
    # 按照你的要求配置命令行参数
    parser.add_argument("--seq-len", type=int, default=131072, help="Sequence length for both Q and KV")
    parser.add_argument("--group-size", type=int, default=16, help="Query Group Size for GQA (Num Q Heads per KV Head)")
    parser.add_argument("--num-heads", type=int, default=128, help="Number of Query Heads")
    parser.add_argument("--warmup", action="store_true", help="Enable warmup")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Disable warmup")
    parser.set_defaults(warmup=True)
    
    args = parser.parse_args()

    # 固定参数设置
    BATCH_SIZE = 1
    D_K = 128
    D_V = 128
    DEVICE = "cuda"
    DTYPE = torch.bfloat16  # FA3 在 H100 上通常使用 BF16 或 FP8

    # 计算 KV Heads (GQA 逻辑)
    NUM_HEADS_Q = args.num_heads
    QUERY_GROUP_SIZE = args.group_size
    if NUM_HEADS_Q % QUERY_GROUP_SIZE != 0:
        raise ValueError(f"num-heads ({NUM_HEADS_Q}) 必须能被 group-size ({QUERY_GROUP_SIZE}) 整除")
    
    NUM_HEADS_KV = NUM_HEADS_Q // QUERY_GROUP_SIZE

    print(f"--- 配置信息 ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Seq Len: {args.seq_len}")
    print(f"Q Heads: {NUM_HEADS_Q}, KV Heads: {NUM_HEADS_KV} (Group Size: {QUERY_GROUP_SIZE})")
    print(f"Head Dim (D_K/D_V): {D_K}/{D_V}")
    print(f"Dtype: {DTYPE}")
    print(f"----------------")

    # 1. 生成输入张量
    # 形状要求通常为: [batch, seqlen, nheads, headdim]
    q = torch.randn(BATCH_SIZE, args.seq_len, NUM_HEADS_Q, D_K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(BATCH_SIZE, args.seq_len, NUM_HEADS_KV, D_K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(BATCH_SIZE, args.seq_len, NUM_HEADS_KV, D_V, device=DEVICE, dtype=DTYPE)

    # 2. 预热 (Warmup)
    if args.warmup:
        print("正在进行预热...")
        for _ in range(3):
            _ = flash_attn_func_v3(q, k, v, causal=False)
        torch.cuda.synchronize()

    # 3. 执行单次调用并计时
    print("正在执行 Flash Attention 3...")
    start_time = time.time()
    
    # 调用 FA3 接口
    # 注意：根据你提供的代码，FA3 接受 q, k, v，可选参数包括 causal, softcap 等
    nvtx.push_range("PROFILE_TARGET_FN")
    output = flash_attn_func_v3(q, k, v, causal=False)
    
    
    torch.cuda.synchronize()
    nvtx.pop_range()
    end_time = time.time()

    # 4. 打印结果
    print(f"执行成功!")
    print(f"输出形状: {output.shape}")
    print(f"单次推理耗时: {(end_time - start_time) * 1000:.3f} ms")

    # 显存占用简单统计
    mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"峰值显存占用: {mem_alloc:.2f} GB")

if __name__ == "__main__":
    run_fmha()