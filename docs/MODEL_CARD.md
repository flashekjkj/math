# Math Tutor 模型说明

原始基座是 Qwen2.5-0.5B-Instruct。原解题 SFT Adapter 为 runs/local-sft-v1，合并模型为 models/math-sft-0.5b；固定200题原始回答正确率24%，接入计算器后系统正确率47.5%。

解释型 SFT 为 runs/analysis-sft-v1，合并模型 models/math-analysis-0.5b，使用954条教师合成分析与列式，验证35/100、固定历史测试68/200。解释型 DPO 为 runs/analysis-dpo-v1，必须配合 models/math-analysis-0.5b；验证35/100、历史回归测试64/200，未显示整体解题收益。

混合多轮 SFT 的配置为 configs/multiturn-sft-v1.json，数据位于 data/versions/multiturn-v1，训练结果以 runs/multiturn-sft-v1/completion.json 为完成依据。直接从原始 Instruct 基座训练，不能与原解题或解释型模型混用 Adapter。

多轮实验使用自然对话提示、单轮数学解释、多轮辅导和通用对话混合。所有assistant轮次参与监督，用户与system token屏蔽，使用原生聊天模板，不静默截断。教师生成和另开请求复核仍属合成标注，不代表人工金标准。

Math23K 的 family 分组是启发式去重，不能证明不存在语义重复。历史200题测试已多次查看，属于回归集。数值校验不代表列式、解释或教学质量正确。服务性能验证见 docs/vLLM评测结果-20261119.md，不将小规模负载结果外推生产环境。
