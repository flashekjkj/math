# GRPO 第一轮训练

GRPO 从 DPO 的 runs/dpo-demo-v1/run_config.json 中记录的 models/math-sft-0.5b 开始；不加载 DPO Adapter。base_model_manifest.json 记录本次基础模型文件 SHA256。DPO 未记录历史模型文件哈希，因此相同路径不能单独证明跨时间文件未被替换。

## 实现

- math_tutor/data/build_rl_data.py：按 DPO 偏好文件逐条抽取完全相同的 prompt，并匹配 processed 数据的答案、ID；排除测试集 family 泄漏。答案和表达式只作为元数据，不加入输入。
- math_tutor/rewards/math_reward.py：使用 numeric-v2 数值评分器，仅明确判对给 1；错误、未解析、冲突和疑似多问给 0。不验证完整推导或题意与算式一致性。
- math_tutor/training/grpo_training.py：TRL GRPOTrainer + 新 LoRA，q_proj/v_proj，rank=8，alpha=16，dropout=0。每题采样4个回答，128 token上限，temperature=0.8，top_p=0.95，beta=0.02，loss_type=grpo，组内奖励归一化。
- 与旧脚本相比：使用固定 train/valid，移除随机重划分及 GSM8K/LaTeX/XML 格式奖励，不关闭梯度检查点。
- 原脚本备份：archive/grpo-before-20261106/grpo_training.py。

## 复现命令（Anaconda Prompt）

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m unittest discover -s tests -v
python -m math_tutor.data.build_rl_data
python -m math_tutor train --stage grpo --config configs/grpo-v1.json --dry-run
python -m math_tutor train --stage grpo --config configs/grpo-v1.json --smoke
python -m math_tutor train --stage grpo --config configs/grpo-v1.json
python -m math_tutor evaluate --model models/math-sft-0.5b --adapter runs/grpo-v1 --data data/processed/valid.jsonl --output results/grpo-v1-valid.jsonl
python -m math_tutor chat --model models/math-sft-0.5b --adapter runs/grpo-v1
```

已有数据目录时不重复执行 build_rl_data。重复训练需复制配置并修改 output_dir；已有评测需换新的 output 文件名，避免覆盖。

## 产出与判读

- runs/grpo-v1/completion.json：仅在训练结束、模型和状态保存成功后生成。
- adapter_model.safetensors、adapter_config.json、tokenizer：使用同一 SFT 基础模型加载。
- run_config.json、environment.lock.txt、code_and_data_manifest.json、base_model_manifest.json：复现证据。
- rollouts.jsonl：逐条采样回答与奖励，可按每4条计算组内差异。
- trainer_state.json、train_results.json、TensorBoard 日志：更新次数、奖励、KL、梯度等。
- results/grpo-v1-valid.*：使用和 SFT、DPO 一样的100题验证集、greedy、192 token上限、numeric-v2评分。

训练200题一轮，每题4个回答，batch=1、梯度累积4，预期200次优化器更新、800条训练采样。验证题不参与参数更新。训练内不在线生成验证，结束后使用统一离线评测。

纯正确性奖励可能稀疏：整组全对或全错时没有正确性优势信号。最初5步 smoke 出现全零奖励及零梯度，只说明流程成功；正式训练仍须检查混合奖励组和非零参数变化。
