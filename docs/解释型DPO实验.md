# 解释型SFT的第一轮DPO实验

基础模型为 `models/math-analysis-0.5b`，输出分析与列式，由程序提取表达式并通过计算器给出答案。模型文件哈希已记录并在训练后核验。

从原解释型SFT训练记录中逐字复用chosen。使用该SFT模型的greedy真实回答作为负例，只保留完整、可执行且计算结果与参考答案不同的回答。已知教师对参考存在争议的题先排除。未新增API生成调用。

偏好训练200对、验证30对，按family划分。偏好验证题也属于原SFT训练池，但未参与DPO参数更新。另以原100题验证集、200题测试集评价生成效果。chosen此前通过教师及自动检查，未逐条人工核验，数值核验不足以证明题意解释正确。

本轮完成25次LoRA更新，batch=1、梯度累积8，相当于200对训练一轮。学习率5e-6，最大总长度512 token。DPO数据过滤改为实际token长度加EOS检查。48个LoRA B张量非零，权重更新已核验。

| 模型与数据 | 正确数 | 计算器系统正确率 | 可执行率 |
|---|---:|---:|---:|
| SFT验证 | 35/100 | 35.0% | 88.0% |
| DPO验证 | 35/100 | 35.0% | 89.0% |
| SFT历史测试 | 68/200 | 34.0% | 88.5% |
| DPO测试 | 64/200 | 32.0% | 88.5% |

逐题变化为 `{"valid": {"gain": 6, "regression": 6, "changed_responses": 43}, "test": {"gain": 3, "regression": 7, "changed_responses": 79}}`。验证基线本轮重新生成，测试基线使用同模型路径、同协议的历史记录。历史基线未保存本轮模型文件哈希，不能据此证明跨时间权重逐位不变。历史测试集已多次查看，属于固定回归集。本轮没有按测试结果选检查点或修改超参数。

该评测衡量模型列式与计算器组成的系统，解释文本语义仍需独立评价。

复现时先将配置output_dir和评测输出改为新的版本，避免覆盖。

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m math_tutor.data.build_analysis_preferences
python -m math_tutor train --stage dpo --config configs/analysis-dpo-v1.json --smoke
python -m math_tutor train --stage dpo --config configs/analysis-dpo-v1.json
python -m math_tutor.evaluation.tool_eval --model models/math-analysis-0.5b --adapter runs/analysis-dpo-v1 --formula-only --system-prompt-file data/versions/teacher-analysis-v1/system_prompt.txt --data data/processed/valid.jsonl --output results/analysis-dpo-retest-valid.jsonl
python -m math_tutor chat --model models/math-analysis-0.5b --adapter runs/analysis-dpo-v1 --calculator-only --system-prompt-file data/versions/teacher-analysis-v1/system_prompt.txt --max-new-tokens 256
```

本轮结论：偏好验证排序准确率约96.7%，但生成验证正确率维持35%，固定测试集相对历史基线从34%降至32%。验证逐题6题改善、6题退步；测试3题改善、7题退步。当前证据未显示整体解题收益。

