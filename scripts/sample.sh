#!/bin/bash

# 设定包含 .ckpt 文件的文件夹路径
ckpt_dir="logs/2024-11-11T09-29-20_teddybear-sdv4/checkpoints"  # 替换成你的ckpt文件夹路径
pretrained_model_path="stable-diffusion-v-1-4-original/sd-v1-4.ckpt"  # 替换成你的预训练模型路径

# 遍历所有 .ckpt 文件
for ckpt_file in "$ckpt_dir"/*[4-9].ckpt "$ckpt_dir"/*1[0-4].ckpt; do
  if [ -f "$ckpt_file" ]; then
    echo "Processing checkpoint: $ckpt_file"
    # 执行 Python 命令
    python sample.py --prompt "<new1> teddybear sitting near a panda" --delta_ckpt "$ckpt_file" --ckpt "$pretrained_model_path"
  fi
done
