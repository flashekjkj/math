# Math Tutor：实验模型说明

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
项目下一阶段方案见 docs/实习项目下一阶段.md。

## 发布状态

研究与学习原型。尚未完成独立教学任务评测、真实用户验收、负载测试和正式部署；未声明生产可用性。
