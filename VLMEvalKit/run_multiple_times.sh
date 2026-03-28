#!/bin/bash

# ===== 环境变量 =====
ENV_VARS=(
  OPENAI_API_KEY=sk-snVxmTtlvlMByjDzdLcWo4IUxQbdorFN3nBDkNf8UyhvZDdY
  OPENAI_API_BASE=https://zjuapi.com/v1/chat/completions
  HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES=4,5,6,7
)

DATA="MMStar"
MODEL="Qwen3-VL-8B-Instruct-iLLaVA"
NPROC=4
RUNS=3

FAIL_COUNT=0

for ((i=1; i<=RUNS; i++))
do
    echo "===== Run $i ====="

    env "${ENV_VARS[@]}" torchrun --nproc-per-node=$NPROC run.py \
        --data $DATA \
        --model $MODEL \
        --verbose \
        --reuse

    if [ $? -ne 0 ]; then
        echo "Run $i failed, but continue..."
        ((FAIL_COUNT++))
    else
        echo "Run $i succeeded"
    fi
done

echo "=========================="
echo "All runs finished"