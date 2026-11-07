# Math Tutor：实验模型说明

## 教师分析标签实验

models/math-analysis-0.5b：954条真实语义分析合成标签SFT，FP16合并，公式由计算器执行。
固定200题测试68/200=34%，100题验证35/100=35%；低于旧SFT＋计算器47.5%，不替换最佳版本。
完整条件、局限与命令见 docs/分析模型训练.md，数据由教师模型生成并自动质检，不代表人工逐条确认。

## 当前关系解释与工具计算版本

完整模型：models/math-relation-tool-0.5b；LoRA：runs/relation-tool-v1；配置：configs/relation-tool.json。
同1000/100题，1 epoch、125次更新，训练约330.7秒，eval_loss约0.1429（训练目标已变化，不可与旧loss直接比较）。
训练标签为算式结构关系说明、列式、calculator JSON调用；每个参考调用经过实际执行验证，不训练模型自己计算最终数字。
运行时严格校验调用与列式是否一致，由受限计算器计算并返回最终答案。不是新增算术专项SFT，也不是自主多步Agent。
关系说明为确定性结构模板，并非人工核验的题意数量关系，不保证提升审题能力。最终数值成绩属于模型与工具组成的系统。
本轮最终测试报告 results/relation-tool-test.summary.json；三组对照 results/relation-tool-comparison.json。

## 用途和模型链

中文小学应用题列式与数值作答的学习实验。基础模型 Qwen/Qwen2.5-0.5B-Instruct；SFT Adapter 为 runs/local-sft-v1；合并模型 models/math-sft-0.5b；DPO Adapter 为 runs/dpo-demo-v1，必须配合该合并模型加载。
当前不是已经验证教学质量的数学教师模型。

## 数据与训练

Math23K 镜像数据，清洗后的当前子集 train=1000、valid=100、test=200。启发式 family 去重和划分，表达式与参考值一致性检查；表达式是否符合题意仍需人工核验。数据和基础模型的许可证、来源及再分发条件必须分别核对，上游训练代码许可证保存在 third_party，不代表数据许可证。
SFT 采用 LoRA q_proj/v_proj、r=8、alpha=16，长度 256，1 epoch；具体配置以对应 run_config.json 为准。
DPO 使用 200/50 合成偏好对、25 次更新。拒绝回答由答案扰动构造，不能代表真实教学偏好。

## 结果

固定 200 题测试：base 数值正确率 11.5%，SFT 24.0%，可解析率 74.0% → 98.5%。仅为当前小样本实验，不能外推其他题型。
100 题验证：合并 SFT 和 DPO 均为 18.0%，未发现 DPO 数值收益。
合并前 Adapter 为 17.0%，与合并模型有 8 条回答差异，原因尚未确定；需核查精度、权重与推理路径。
当前评分 numeric-v2，仅验证最终数字；可能出现答案正确但推导错误、单位错误或格式不可解析。

## 工程与复现

代码来源与适配记录位于 third_party/provenance.json。独立项目无需运行相邻 MedicalGPT。
训练目录保留配置、数据/代码哈希及环境快照。已有历史评测的模型权重与运行环境快照不足以证明逐位可复现，应在新实验增加权重标识与环境记录。
离线审计：python -m math_tutor.evaluation.project_audit --output reports/new-audit
当前使用方案见 docs/使用说明.md。

## 发布状态

研究与学习原型。尚未完成独立教学任务评测、真实用户验收、负载测试和正式部署；未声明生产可用性。
