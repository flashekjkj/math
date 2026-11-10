# PPO 第一轮训练与评测

PPO 从 `models/math-sft-0.5b` 开始训练。该路径与 DPO、GRPO 的运行记录一致，模型文件 SHA256 与 GRPO 留存记录逐项匹配。DPO 留存了模型路径。

复用与 DPO、GRPO 相同的 200 道训练题。每题采样一次，每 4 条采样执行 2 次 PPO 更新，共 100 次优化器更新，训练耗时 109.6 秒。规则奖励沿用 GRPO 的数值评分器，另有权重为 0.02 的参考策略 KL 惩罚。PPO 采样使用温度 0.8 的完整分布，GRPO 每题采样 4 次并使用 top-p 0.95。

策略通过 LoRA 更新，价值头与策略共享 Qwen 主干。价值头估计 token 前缀的回报，GAE 计算优势，PPO 在每批采样上使用裁剪后的策略和价值目标。参考策略由禁用 LoRA 后的 SFT 模型提供。[PPO 论文](https://arxiv.org/abs/1707.06347) · [GAE 论文](https://arxiv.org/abs/1506.02438)

正式训练采样获得正确奖励 25/200 条，86/100 步梯度非零。最终有 48 个非零 LoRA B 张量，价值头权重 L2 范数为 0.019466，两者均从零初始化后发生变化。峰值 CUDA 已分配内存约 1.30 GiB。

| 模型 | 确认正确 | 数值正确率 | 解析率 |
|---|---:|---:|---:|
| SFT | 18/100 | 18.0% | 99.0% |
| DPO | 18/100 | 18.0% | 99.0% |
| GRPO | 18/100 | 18.0% | 99.0% |
| PPO | 19/100 | 19.0% | 100.0% |

评测统一使用同一 100 道验证题、同一提示词、greedy 解码、192 token 上限和同一评分器。相对 SFT，PPO 有 1 题新判对、0 题失去正确判定、19 条回答文本发生变化。各方法的训练采样预算不同。

唯一新增判对的题是 `train:3752`，要求计算 50 个零件中合格率 98% 对应的合格数量。SFT 已给出正确的 `50 × 98% = 49`，随后追加了冲突结论 `62.5`。PPO 在给出 `49` 后结束。这 1 分的增益来自消除重复结论造成的冲突。

在 Anaconda Prompt 中运行以下命令。重跑训练需在配置中指定新的输出目录，重跑评测需使用新的结果文件名。

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m math_tutor train --stage ppo --config configs/ppo-v1.json --smoke
python -m math_tutor train --stage ppo --config configs/ppo-v1.json
python -m math_tutor evaluate --model models/math-sft-0.5b --adapter runs/ppo-v1 --data data/processed/valid.jsonl --output results/ppo-v1-valid.jsonl
python -m math_tutor chat --model models/math-sft-0.5b --adapter runs/ppo-v1
```

`runs/ppo-v1` 保存 LoRA Adapter、价值头、采样记录、更新指标和模型哈希。推理加载 SFT 模型和 LoRA Adapter。原有 RLOO 使用 `--stage rloo`。
