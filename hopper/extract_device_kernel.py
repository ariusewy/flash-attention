#!/usr/bin/env python3
import csv
import glob
import os
import re

input_dir = "nsys_reports"
output_file = "device_kernel_summary.csv"

# 输出表头
header = ["Heads", "SeqLen", "Total_Time_us", "Instances", "Avg_us", "Med_us", "Min_us", "Max_us", "StdDev_us"]
data = [header]

print("开始从所有 kernel_stats.csv 中提取 device_kernel 时间统计...")
print(f"搜索目录: {input_dir}")

processed = 0
# 修正后的 glob 模式（兼容 nsys 自动添加的后缀）
for csv_file in sorted(glob.glob(f"{input_dir}/*_kernel_stats.csv*")):
    basename = os.path.basename(csv_file)
    print(f"  发现文件: {basename}")
    
    # 从文件名解析 Heads 和 SeqLen
    match = re.search(r'fmha_profile_h(\d+)_s(\d+)', basename)
    if not match:
        print(f"    跳过（文件名格式不匹配）: {basename}")
        continue
    heads = int(match.group(1))
    seqlen = int(match.group(2))
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            # 匹配包含 "device_kernel" 的主 Flash Attention kernel
            if len(row) > 8 and "device_kernel" in row[-1]:
                try:
                    total_ns = float(row[1])
                    instances = int(row[2])
                    avg_ns = float(row[3])
                    med_ns = float(row[4])
                    min_ns = float(row[5])
                    max_ns = float(row[6])
                    std_ns = float(row[7])
                    
                    # ns → us 并保留三位小数
                    total_us = total_ns / 1000
                    avg_us = avg_ns / 1000
                    med_us = med_ns / 1000
                    min_us = min_ns / 1000
                    max_us = max_ns / 1000
                    std_us = std_ns / 1000
                    
                    data.append([
                        heads,
                        seqlen,
                        round(total_us, 3),
                        instances,
                        round(avg_us, 3),
                        round(med_us, 3),
                        round(min_us, 3),
                        round(max_us, 3),
                        round(std_us, 3)
                    ])
                    processed += 1
                    print(f"  ✓ Heads={heads:3d} | SeqLen={seqlen:4d} → {total_us:8.3f} us")
                except Exception as e:
                    print(f"  ✗ 解析失败 {basename}: {e}")
                break  # 每个 CSV 只取第一个匹配的 device_kernel

# 写入汇总 CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"\n提取完成！共处理 {processed} 个配置。")
print(f"结果已保存至: {output_file}")
if processed == 0:
    print("提示：若仍为 0，请确认 nsys_reports/ 目录下存在以 _kernel_stats.csv* 结尾的文件。")
else:
    print("您可直接用 Excel 或 LibreOffice 打开查看。")