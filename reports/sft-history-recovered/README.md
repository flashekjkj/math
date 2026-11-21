# 找回的历史 SFT 与 API 对照证据

恢复日期：2026-11-20。原文件位于 F:/codex_content/2026-10-26/agent-2/outputs/，原始文件未修改。该目录名不是实验发生日期的证明，具体训练条件以各文件记录为准。

- ablation-comparison.json：0.5B 基线、2000条扩数、1000条2 epoch；对应 config、raw 与 calculator 逐题文件一并保留。
- model-scale-comparison.json：1.5B 2000条 QLoRA，与0.5B同数据量对照。
- api-summary.json / api-predictions.jsonl：deepseek-v4.1-flash，200题168题正确，84%。
- provenance.json：每份源文件路径与SHA256、测试集哈希、逐题ID/题目/参考值匹配检查。

本次重新计数确认全部逐题文件各200条，模型原始正确数46/44/58、工具正确数100/97/101，API正确168，测试ID、题目及参考值与当前200题一致。未重新训练、推理或调用API；原权重与历史环境未复验。

当前项目另有3000条准备数据，但尚未找到相应训练/评测完成记录；不要将2000条实验误标为3000条。API返回模型均为deepseek-v4.1-flash，没有找到Qwen3.8测试证据。
