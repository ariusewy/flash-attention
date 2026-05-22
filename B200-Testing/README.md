# B200 FA4 Profiling — 服务器部署与逐步测试指南

> Author: ywangmu (HKUST)  
> Last updated: 2026-05-22

本文档说明如何将 B200 Testing 脚本包上传到 B200 服务器，逐步探测环境、运行 FA4 case、采集 NCU Report，并汇总为 `ALL_RESULTS.csv`。

**不需要从源码编译 FlashAttention**；服务器上通过 `pip install flash-attn-4` 安装 FA4 即可。

---

## 目标工作流

```
本地打包 scripts/  →  上传到 B200  →  环境探测  →  环境搭建（如需）
    →  correctness / perf  →  NCU profiling  →  parse  →  collect_results
```

最终交付物：`<outdir>/ALL_RESULTS.csv`（含 latency、L2、DRAM、TMA traffic 等）。

---

## 1. 打包与上传

### 1.1 需要打包的内容

请打包整个 `fa4_latest/scripts/` 目录（**不是**仅 `B200-Testing/`），因为 benchmark 与 NCU 脚本在同级目录：

```
scripts/                          ← 解压后的工作根目录 ($SCRIPTS)
├── bench_fa4_simfa.py            # FA4 benchmark（correctness / perf / run_once）
├── ncu_profile_fa4.sh            # 单 shape NCU profiling
├── parse_ncu_report.py           # .ncu-rep → summary.json / CSV
├── B200_probe.sh                 # 环境探测
├── B200_setup_fa4.sh             # conda 环境一键搭建
└── B200-Testing/
    ├── run_b200_all.sh           # 一键编排（smoke / full）
    ├── collect_results.py        # 汇总 NCU + perf → ALL_RESULTS.csv
    └── README.md                 # 本文件
```

### 1.2 本地打包

在开发机上执行：

```bash
tar -czf b200_fa4_testing.tar.gz \
  -C /home/wangya/Proj/Ramulator2ForAI/fa4_latest scripts
```

或使用 rsync 直接同步：

```bash
rsync -avz --progress \
  /home/wangya/Proj/Ramulator2ForAI/fa4_latest/scripts/ \
  <b200_user>@<b200_host>:~/fa4_scripts/
```

### 1.3 服务器解压

```bash
scp b200_fa4_testing.tar.gz <b200_user>@<b200_host>:~
ssh <b200_user>@<b200_host>

mkdir -p ~/fa4_scripts
tar -xzf ~/b200_fa4_testing.tar.gz -C ~/fa4_scripts
cd ~/fa4_scripts/scripts    # 下文 $SCRIPTS 均指此目录
```

---

## 2. 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| GPU | B200 (SM 10.0) | `nvidia-smi` 应显示 Blackwell |
| Driver | ≥ R570 | 支持 sm_100 |
| CUDA | ≥ 12.8 | Blackwell PTX 支持 |
| Nsight Compute | ≥ 2025.1 | `ncu` 在 PATH 中 |
| Conda env | `b200_fa3` | `torch ≥ 2.6`，`flash-attn-4` |
| sudo / 计数器权限 | `--full` NCU 必需 | DRAM / TMA byte counter 需要 |

FA4 安装方式（无需编译源码）：

```bash
bash $SCRIPTS/B200_setup_fa4.sh
# 或仅验证：bash $SCRIPTS/B200_setup_fa4.sh --verify
```

---

## 3. 逐步操作（推荐顺序）

以下命令均在 B200 服务器上、于 `$SCRIPTS` 目录下执行。  
可通过 `export GPU_ID=0` 选择 GPU；默认 `GPU_ID=0`。

### Step 1 — 环境探测

```bash
cd ~/fa4_scripts/scripts
bash B200_probe.sh | tee B200-Testing/env_probe_$(date +%Y%m%d).txt
```

**期望输出：**

- `SM arch: 10.0`（Blackwell）
- `flash_attn: True  source=flash_attn_interface`
- `ncu: ...` 版本信息
- `RestrictProfiling = 0` 或说明需 sudo

若 FA4 未安装，先运行 Step 1b。

### Step 1b — 环境搭建（首次部署）

```bash
bash B200_setup_fa4.sh
bash B200_probe.sh    # 再次确认
```

### Step 2 — Correctness 检查

确认 FA4 输出与 PyTorch SDPA 参考一致：

```bash
conda run -n b200_fa3 python bench_fa4_simfa.py \
  --mode correctness --cases minimal

# GQA 形状（Llama3-8B 风格）
conda run -n b200_fa3 python bench_fa4_simfa.py \
  --mode correctness --cases llama3_8b
```

全部 PASS 后再进行 perf / NCU。

### Step 3 — Performance（Python 计时，无需 sudo）

测量 `fwd_ms` / TFLOPS，可与 NCU `duration_us` 交叉验证：

```bash
mkdir -p B200-Testing/results/manual_run

conda run -n b200_fa3 python bench_fa4_simfa.py \
  --mode perf --cases small --no-backward \
  -o B200-Testing/results/manual_run/perf_small.json

conda run -n b200_fa3 python bench_fa4_simfa.py \
  --mode perf --cases ncu_sweep --no-backward \
  -o B200-Testing/results/manual_run/perf_ncu_sweep.json
```

### Step 4 — NCU Profiling（单 shape，需 sudo）

对每个 shape 单独 profile。示例：Llama3-8B，seqlen=2048，GQA (H_Q=32, H_KV=8)：

```bash
OUTDIR=B200-Testing/results/manual_run/ncu_s2048_hq32_hkv8_d128 \
GPU_ID=0 \
sudo -E bash ncu_profile_fa4.sh \
  --seqlen 2048 \
  --heads 32 \
  --heads-kv 8 \
  --headdim 128 \
  --batch 1 \
  --no-backward \
  --full
```

**输出目录**（由 `OUTDIR` 指定）：

```
ncu_s2048_hq32_hkv8_d128/
├── env.txt
├── profile.ncu-rep          ← 原始 NCU 报告（可带回本地）
├── profile.csv
├── summary.json             ← 解析后的关键指标
├── profile_calib.csv
├── app_stdout.txt
├── ncu_run.log
└── summary.txt
```

#### 预设 NCU shape 列表（`--full` 模式）

| # | seqlen | H_Q | H_KV | headdim | 说明 |
|---|--------|-----|------|---------|------|
| 1 | 512 | 8 | 8 | 128 | MHA smoke |
| 2 | 1024 | 32 | 8 | 128 | Llama3-8B short |
| 3 | 2048 | 32 | 8 | 128 | Llama3-8B |
| 4 | 4096 | 64 | 8 | 128 | Llama3-70B-like |
| 5 | 8192 | 128 | 8 | 128 | Llama3-405B-like |

批量跑 5 个 shape 的示例：

```bash
RESULT_ROOT=B200-Testing/results/manual_run
mkdir -p "$RESULT_ROOT"

run_one() {
  local s=$1 hq=$2 hkv=$3 d=$4
  local label="ncu_s${s}_hq${hq}_hkv${hkv}_d${d}"
  echo "=== Profiling $label ==="
  OUTDIR="$RESULT_ROOT/$label" GPU_ID=0 \
  sudo -E bash ncu_profile_fa4.sh \
    --seqlen "$s" --heads "$hq" --heads-kv "$hkv" \
    --headdim "$d" --batch 1 --no-backward --full
}

run_one 512  8   8 128
run_one 1024 32  8 128
run_one 2048 32  8 128
run_one 4096 64  8 128
run_one 8192 128 8 128
```

### Step 5 — 重新解析已有 .ncu-rep（可选）

若只需重新提取指标（例如更新了 `parse_ncu_report.py`），无需重跑 NCU：

```bash
conda run -n b200_fa3 python parse_ncu_report.py \
  B200-Testing/results/manual_run/ncu_s2048_hq32_hkv8_d128/profile.ncu-rep \
  -o B200-Testing/results/manual_run/ncu_s2048_hq32_hkv8_d128/summary.json \
  --csv B200-Testing/results/manual_run/ncu_s2048_hq32_hkv8_d128/profile_calib.csv \
  --pretty
```

### Step 6 — 汇总结果

将 Step 3 perf JSON 与 Step 4 NCU 目录合并为一张表：

```bash
conda run -n b200_fa3 python B200-Testing/collect_results.py \
  B200-Testing/results/manual_run \
  -o B200-Testing/results/manual_run/ALL_RESULTS.csv
```

检查输出：

```bash
column -s, -t B200-Testing/results/manual_run/ALL_RESULTS.csv | head
```

### Step 7 — 打包结果回本地

```bash
tar -czf b200_results_$(date +%Y%m%d).tar.gz \
  -C B200-Testing/results manual_run
```

---

## 4. 一键模式（可选）

熟悉逐步流程后，可用 master 脚本代替 Step 2–6。

### Smoke（~5 min，无需 sudo）

```bash
bash B200-Testing/run_b200_all.sh --smoke
# 输出：B200-Testing/results/b200_run_<timestamp>/
```

### Full（~60–90 min，需 sudo）

```bash
sudo bash B200-Testing/run_b200_all.sh \
  --full \
  --outdir /mnt/nvme3n1/b200_results \
  --gpu 0
```

可选参数：`--env b200_fa3`、`--lock-mhz 1600`（锁频以提高可重复性）。

---

## 5. 输出目录结构

```
<outdir>/
├── env_report.txt              # Step 1 环境快照（一键模式）
├── correctness.log
├── perf_small.json             # Python fwd_ms
├── perf_ncu_sweep.json         # full 模式
├── perf_llama3_all.json        # full 模式
├── perf.log
├── ncu_s512_hq8_hkv8_d128/
│   ├── profile.ncu-rep         # 原始 NCU 报告
│   ├── summary.json            # 解析指标
│   └── profile_calib.csv
├── ncu_s2048_hq32_hkv8_d128/
│   └── ...
├── ncu_run.log
├── ncu_parse.log
├── collect.log
├── run_summary.txt             # 各 step pass/fail（一键模式）
└── ALL_RESULTS.csv             # ← 主交付物
```

---

## 6. 采集指标说明

### 6.1 Latency

| CSV 字段 | NCU / 来源 | 说明 |
|----------|------------|------|
| `duration_us` | `gpu__time_duration.sum` | Kernel 硬件耗时（µs） |
| `fwd_ms` | Python `time.perf_counter` | 整次 forward 调用（ms） |

### 6.2 HBM (DRAM)

| CSV 字段 | NCU metric | 说明 |
|----------|------------|------|
| `dram_bytes_read` | `dram__bytes_read.sum` | HBM 读字节 |
| `dram_bytes_write` | `dram__bytes_write.sum` | HBM 写字节 |
| `dram_throughput_pct` | `dram__throughput.*` | HBM 带宽利用率 |

> 需要 `sudo` 或 `RestrictProfiling=0`，否则为空。

### 6.3 L2 Cache (LTS)

| CSV 字段 | NCU metric | 说明 |
|----------|------------|------|
| `lts_sectors` | `lts__t_sectors.sum` | L2 访问 sector 总数 |
| `lts_hit_rate_pct` | `lts__t_sector_hit_rate.pct` | L2 hit rate |
| `lts_miss_sectors` | `lts__t_sectors_lookup_miss.sum` | L2 miss sector |

1 sector ≈ 32 B。L2 指标为**整体** traffic，不按 TMA/LSU 拆分。

### 6.4 TMA Traffic（GMEM ↔ SMEM）

| CSV 字段 | NCU metric | 说明 |
|----------|------------|------|
| `tma_ld_bytes` | `l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum` | TMA load 字节（Q/K/V） |
| `tma_ld_sectors` | `l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum` | TMA load sectors |
| `tma_st_bytes` | `l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st.sum` | TMA store 字节（O） |
| `tma_st_sectors` | `l1tex__m_l1tex2xbar_write_sectors_mem_global_op_tma_st.sum` | TMA store sectors |
| `tma_pipe_read_bytes` | `l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum` | TMA pipe 总读 |
| `tma_pipe_write_bytes` | `l1tex__m_l1tex2xbar_write_bytes_pipe_tma.sum` | TMA pipe 总写 |
| `tma_ld_inst` | `sm__sass_inst_executed_op_tma_ld.sum` | TMA load 指令数 |
| `tma_st_inst` | `sm__sass_inst_executed_op_tma_st.sum` | TMA store 指令数 |
| `tma_requests` | `l1tex__tmain_requests.sum` | TMA 请求数 |
| `tma_pipe_util_pct` | `sm__pipe_tma_cycles_active.*` | TMA pipe 利用率 |

> TMA byte counter 同样需要 sudo。旧版 `.ncu-rep` 若未采集这些 metric，需重跑 NCU。

### 6.5 TMEM

NCU **不提供** TMEM byte-level traffic counter。TMEM 行为通过 simulator trace / analytical model 分析，不在本测试包的硬件指标范围内。

### 6.6 Pipeline / Compute

| CSV 字段 | 说明 |
|----------|------|
| `tensor_pipe_util_pct` | Tensor Core (UMMA) 利用率 |
| `stall_barrier_pct` | Barrier stall 比例 |
| `stall_long_scoreboard_pct` | Memory dependency stall |
| `fwd_tflops` | Python perf 模式计算的 TFLOPS |

---

## 7. 与论文图表的对应

| 图表 | 使用字段 | 来源 |
|------|----------|------|
| Latency bar chart | `duration_us`, `fwd_ms` | NCU + perf |
| Tensor Core util | `tensor_pipe_util_pct` | NCU |
| Memory traffic (HBM/L2/TMA) | `dram_*`, `lts_*`, `tma_*` | NCU |
| Pipeline Gantt | — | Ramulator2 `GANTT_LOG`（非本包） |
| GQA scaling | `fwd_tflops` vs seqlen | perf JSON |

---

## 8. 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `SM arch: 9.0` | 跑在 H100/H800 上 | 检查 `CUDA_VISIBLE_DEVICES` |
| `flash_attn: False` | 未装 FA4 | `bash B200_setup_fa4.sh` |
| `ERR_NVGPUCTRPERM` | 无 perf counter 权限 | `sudo -E bash ncu_profile_fa4.sh ...` |
| `dram_bytes_read` / `tma_ld_bytes` 为空 | 未 sudo | 同上 |
| NCU 很慢 / timeout | `--full` 指标多 | 单 shape 预留 5–10 min |
| `FA4 FAILED: num_heads_kv` | 误用 FA3 fallback | 确认 `flash_attn_interface` 可用 |
| GQA shape 结果像 MHA | 未传 `--heads-kv` | NCU 命令需 `--heads-kv 8` |
| `collect_results` 无 NCU 行 | 目录名不符 | NCU 目录须为 `ncu_s*_hq*_hkv*_d*` |

---

## 9. 快速命令索引

```bash
# 环境
bash B200_probe.sh
bash B200_setup_fa4.sh

# Correctness
conda run -n b200_fa3 python bench_fa4_simfa.py --mode correctness --cases minimal

# Perf
conda run -n b200_fa3 python bench_fa4_simfa.py --mode perf --cases ncu_sweep \
  --no-backward -o perf_ncu_sweep.json

# NCU 单 shape
OUTDIR=./ncu_out GPU_ID=0 sudo -E bash ncu_profile_fa4.sh \
  --seqlen 2048 --heads 32 --heads-kv 8 --headdim 128 --no-backward --full

# 解析
conda run -n b200_fa3 python parse_ncu_report.py ncu_out/profile.ncu-rep \
  -o ncu_out/summary.json --pretty

# 汇总
conda run -n b200_fa3 python B200-Testing/collect_results.py . -o ALL_RESULTS.csv

# 一键
bash B200-Testing/run_b200_all.sh --smoke
sudo bash B200-Testing/run_b200_all.sh --full --outdir /path/to/out
```
