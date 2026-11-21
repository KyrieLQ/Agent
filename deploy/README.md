# GPU 服务器部署

发布的代码是项目侧的任务入口、轨迹筛选、训练启动脚本和 vLLM 服务配置。模型、轨迹和上游源码由部署者自行准备。训练依赖与本机的 PyTorch、CUDA 和 GPU 架构匹配。

## 1. 准备依赖

- Linux、Bash、NVIDIA 驱动和 Python 3.12。
- 服务环境参考项目使用的 vLLM 0.18.1。训练环境按 [MERA-Evolve 的 τ²-bench 配方](https://github.com/yh-yao/MERA-Evolve/tree/main/experiments/tau-2) 准备。
- 准备 [MERA-Evolve](https://github.com/yh-yao/MERA-Evolve)、[router-skills-evolve](https://github.com/zeyuyuyu/router-skills-evolve) 和 [τ²-bench](https://github.com/sierra-research/tau2-bench)。在可导入 `tau2`、`pydantic` 和 LiteLLM 的 runner 环境运行任务入口。

在项目根目录设置路径。使用实际目录替换示例值。

```bash
export MERA_ROOT=/path/to/MERA-Evolve
export TAU2_WORKSPACE=/path/to/router-skills-evolve
export TAU2_STAGE2_ROOT="$TAU2_WORKSPACE/tau2_stage2"
export TAU2_DATA_DIR=/path/to/tau2-bench/data
export TRAIN_VENV=/path/to/training-venv
export VLLM_BIN=/path/to/serving-venv/bin/vllm
export PYTHONPATH="$MERA_ROOT/experiments/tau-2:$TAU2_STAGE2_ROOT/code:${PYTHONPATH:-}"
```

## 2. 启动模型服务

分别在两个终端执行。示例把 2B 客服放到 GPU 0，把 27B GPTQ INT4 模拟用户放到 GPU 1、2。两边只监听本机。

```bash
AGENT_GPUS=0 bash deploy/serve.sh agent /path/to/agent-checkpoint
SIM_GPUS=1,2 bash deploy/serve.sh simulator /path/to/simulator-checkpoint
```

如显存不足，可以降低 `AGENT_CONTEXT` 或 `SIM_CONTEXT`；归档配置使用的上下文上限分别是 40,960 和 32,768 token。测试前可用 `curl -fsS http://127.0.0.1:8260/v1/models` 和端口 8261 的对应地址确认服务就绪。下面的任务 ID 由部署者从自己的 benchmark 安装中选取。

```bash
python src/run_task.py telecom "$TASK_ID" --with-playbook --out runs/example.json
```

## 3. 启动训练

教师也可用 `agent` 模式启动。通过 `AGENT_GPUS` 选择分卡，通过 `AGENT_MEMORY_FRACTION` 配置显存，通过 `AGENT_PORT` 设置独立端口。准备了训练侧轨迹后，先用 `training/prepare_teacher.py` 按官方评分与训练 ID 清单筛选，再用上游 `tau2_evolve.traces_to_sft` 转成 SFT parquet。GRPO parquet 由上游 `tau2_evolve.prepare_grpo_data` 从训练侧轨迹生成，模拟用户地址指向上面的服务。

`TEACHER_JSONL` 每行是任务入口输出的一条教师轨迹，`TRAIN_IDS` 是仅包含训练任务的 JSON 数组，元素格式为 `[domain, task_id]`。以下命令的检查点和产物路径使用绝对路径，base 路径保留 `Qwen3.5` 以启用上游相应的兼容配置。

```bash
python training/prepare_teacher.py --traces "$TEACHER_JSONL" --train-ids "$TRAIN_IDS" --output "$VISIBLE_JSONL"
python -m tau2_evolve.traces_to_sft --traces "$VISIBLE_JSONL" --output "$SFT_PARQUET"
python -m tau2_evolve.prepare_grpo_data --traces "$VISIBLE_JSONL" --output "$GRPO_PARQUET" \
  --user-base-url http://127.0.0.1:8261/v1 --no-user-thinking --max-steps 60
MODEL_PATH="$BASE_CHECKPOINT" SFT_DATA="$SFT_PARQUET" SFT_OUTPUT_DIR=/path/to/sft-adapter \
  SFT_GPU=0 bash training/train.sh sft
MODEL_PATH="$BASE_CHECKPOINT" TRAIN_FILE="$GRPO_PARQUET" LORA_ADAPTER_PATH=/path/to/sft-adapter \
  OUTPUT_DIR=/path/to/grpo TRAIN_GPU=0 ROLLOUT_GPU=3 bash training/train.sh grpo
```

训练完成后，用训练环境将输出的 LoRA adapter 与对应 base checkpoint 合并，再用服务脚本启动。

```bash
"$TRAIN_VENV/bin/python" training/merge_adapter.py \
  --mera-root "$MERA_ROOT" --base "$BASE_CHECKPOINT" --adapter "$GRPO_ADAPTER" \
  --output "$MERGED_CHECKPOINT"
bash deploy/serve.sh agent "$MERGED_CHECKPOINT"
```

## 4. Qwen3.5-4B

4B 的 SFT 和 GRPO 均使用两卡 FSDP。GRPO 将更新后的 LoRA 合入同步权重，通过 BF16 权重同步把训练策略传给独立的采样卡。训练入口会将两份 FSDP 分片导出为 PEFT adapter。

```bash
MODEL_PATH="$BASE_4B" SFT_DATA="$SFT_PARQUET" SFT_OUTPUT_DIR="$SFT_ADAPTER" \
  SFT_GPU=0,1 bash training/train.sh sft-4b
```

在独立终端把 27B 模拟用户放在 GPU 3、4，给 GRPO 的训练和采样留出 GPU 0、1、2。

```bash
SERVING_PROFILE=4b SIM_GPUS=3,4 SERVE_MAX_SEQS=8 \
  bash deploy/serve.sh simulator /path/to/simulator-checkpoint
```

```bash
MODEL_PATH="$BASE_4B" TRAIN_FILE="$GRPO_PARQUET" LORA_ADAPTER_PATH="$SFT_ADAPTER" \
  OUTPUT_DIR="$GRPO_OUTPUT" TRAIN_GPU=0,1 ROLLOUT_GPU=2 bash training/train.sh grpo-4b
```

训练环境中的 Qwen3.5 Triton 兼容目录通过 `QWEN35_TRAIN_TRITON_OVERLAY` 指定。GRPO 使用独立源码目录时，通过 `MERA_GRPO_ROOT` 指定。将 `$GRPO_OUTPUT/final_adapter_path.txt` 里的路径传给上面的合并命令，再启动 4B 客服并复用模拟用户服务。

```bash
SERVING_PROFILE=4b AGENT_GPUS=0 bash deploy/serve.sh agent "$MERGED_CHECKPOINT"
python src/run_task.py retail "$TASK_ID" --with-playbook \
  --agent-temperature 0.5 --agent-top-p 0.98 --out runs/retail.json
```
