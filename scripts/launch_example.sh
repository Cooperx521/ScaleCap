#!/bin/bash


LLM_MODEL_PATH="/path/models--Qwen--Qwen2-72B-Instruct/"
LLM_SERVE_CMD="vllm serve $LLM_MODEL_PATH --trust-remote-code --tensor-parallel-size=2 --pipeline-parallel-size=1 --gpu_memory_utilization=0.95 --served-model-name=qwen2-llm --host 0.0.0.0 --max-model-len 32000"

CUDA_VISIBLE_DEVICES=0,1 nohup bash -c "$LLM_SERVE_CMD --port $((21000))" > /dev/null 2>&1 &
VLLM_PIDS_LLM=$!
VLLM_PORTS_LLM=$((21000))
echo "Started vllm API on GPU, listening on port $((21000))"


VLM_MODEL_PATH="/path/models--Qwen--Qwen2.5-VL-7B-Instruct"
VLM_SERVE_CMD="vllm serve $VLM_MODEL_PATH --trust-remote-code --tensor-parallel-size=1 --pipeline-parallel-size=1 --gpu_memory_utilization=0.95 --served-model-name=qwen2-vlm --host 0.0.0.0 --max-model-len 32000"

CUDA_VISIBLE_DEVICES=2 nohup bash -c "$VLM_SERVE_CMD --port $((31000))" > /dev/null 2>&1 &
VLLM_PIDS_VLM=$!
VLLM_PORTS_VLM=$((31000))
echo "Started vllm API on GPU, listening on port $((31000))"



wait_times=0
while ! curl -s http://127.0.0.1:${VLLM_PORTS_LLM} > /dev/null; do
    wait_times=$((wait_times + 1))
    if [ $wait_times -ge 50 ]; then
        echo "Reaching maximum waiting time for vllm, restarting..."
        kill ${VLLM_PIDS_LLM} 2>/dev/null
        sleep 10
        VLLM_PORTS_LLM=$((VLLM_PORTS_LLM + 100))
        CUDA_VISIBLE_DEVICES=0,1 nohup bash -c "$LLM_SERVE_CMD --port ${VLLM_PORTS_LLM}" > /dev/null 2>&1 &
        VLLM_PIDS_LLM=$!
        echo "Restarted vllm API on GPU, listening on port ${VLLM_PORTS_LLM}"
        wait_times=0
    fi
    echo "Waiting for vllm API on VLLM_PORTS_LLM port ${VLLM_PORTS_LLM} to start..."
    sleep 20
done


wait_times=0
while ! curl -s http://127.0.0.1:${VLLM_PORTS_VLM} > /dev/null; do
    wait_times=$((wait_times + 1))
    if [ $wait_times -ge 50 ]; then
        echo "Reaching maximum waiting time for vllm, restarting..."
        kill ${VLLM_PIDS_VLM} 2>/dev/null
        sleep 10
        VLLM_PORTS_VLM=$((VLLM_PORTS_VLM + 100))
        CUDA_VISIBLE_DEVICES=2 nohup bash -c "$VLM_SERVE_CMD --port ${VLLM_PORTS_VLM}" > /dev/null 2>&1 &
        VLLM_PIDS_VLM=$!
        echo "Restarted vllm API on GPU, listening on port ${VLLM_PORTS_VLM}"
        wait_times=0
    fi
    echo "Waiting for vllm API on VLLM_PORTS_VLM port ${VLLM_PORTS_VLM} to start..."
    sleep 20
done


CUDA_VISIBLE_DEVICES=3 python example.py \
    --llm-port "$VLLM_PORTS_LLM" \
    --vlm-port "$VLLM_PORTS_VLM"


ps aux | grep vllm
pgrep -f vllm
pkill -9 -f vllm
pkill -9 -f qwen2-vl
