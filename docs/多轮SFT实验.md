# 混合多轮 SFT 实验与测试方法

## 目标和运行路径

提升模型自身的多轮指令遵循与辅导体验：区分闲聊与解题、记住上下文、修改题目条件、错误纠正、必要澄清，以及只要提示时不公布答案。训练直接从 models/qwen-0.5b 开始，使用新 LoRA；不在旧答案格式Adapter上叠加。

统一系统提示保存在 data/versions/multiturn-v1/system_prompt.txt。保留原有严格解题 SYSTEM 供历史评测复现；自然对话 CHAT_SYSTEM 已更新。纯模型测试显式指定提示文件，配合 --keep-history，避免自动路由、训练题参考和禁用Adapter影响结果。

## 数据

生成器：math_tutor/data/multiturn_data.py。主模型 qwen3.8-flash，仅明确额度耗尽时切换用户授权的 qwen3.8-2.4t-a95b。先生成恰好3组问答，再独立请求审核；最后可运行 audit_multiturn 做上下文状态二次审核。开源权重或许可证属性没有从模型名称推断，记录服务请求和返回的实际模型标识。

单轮数据复用通过质检的教师分析题，补入由受限计算器计算的答案；多轮数学数据覆盖提示、错误纠正、条件修改、澄清、追问、任务切换；通用对话包含学习计划、爱好记忆和自然交流。数据比例以最终manifest为准，目标为50%单轮、30%多轮数学、20%通用多轮。候选不足时宁可减少规模，不加入未确认样本。

训练/验证按family隔离，排除旧valid/test题族。新验证题族取自旧SFT训练池，所以对本次从原始基座开始的新训练未见，对旧历史模型并非未见。闲聊按独立生成任务划分，不保证主题不重叠。合成复核不等于人工金标准。

## 训练与复现

```powershell
conda activate math
cd E:\desktop\LLM_Project\math-tutor
python -m math_tutor.training.multiturn_sft --config configs/multiturn-sft-v1.json
```

输出目录非空会拒绝覆盖；复现时复制配置并修改output_dir。数据生成密钥通过 DASHSCOPE_API_KEY 进程环境提供，不写代码或配置。环境快照、数据和代码哈希、Trainer状态、Adapter及完成标识保存在run目录。

原生聊天模板分词，对全部assistant回答与结束标记监督，system/user部分标签为-100。长度超过1024直接报错，不悄悄丢弃最后追问。训练FP16、q_proj/v_proj LoRA r=8/alpha=16、batch=1/累积8、学习率1e-4、1epoch、seed42。没有使用固定多轮测试结果挑选检查点或调参。

## 自动多轮回归

```powershell
python -m math_tutor.evaluation.multiturn_eval --output results/multiturn-base-retest.jsonl
python -m math_tutor.evaluation.multiturn_eval --adapter runs/local-sft-v1 --output results/multiturn-old-sft-retest.jsonl
python -m math_tutor.evaluation.multiturn_eval --adapter runs/multiturn-sft-v1 --output results/multiturn-sft-retest.jsonl
```

每个输出文件必须使用新名字。24段对话、每段3轮，共72次生成；greedy，最多128新token。同一用例与提示，每个模型携带自己前几轮的真实回答，属于闭环测试，不用标准答案替换历史。记录suite哈希、Adapter哈希、逐轮原文、截断和自动检查。输入用例由程序固定编写，未根据本次输出修改。

规则检查包括数值出现、只输出数字、提示不泄露答案、纠错关键词、闲聊无数学模板和爱好记忆。它们是代理指标：出现正确数字不等于最终结论正确，出现“正确”等词也不能证明纠错正确。必须人工逐轮确认；不能将这些规则通过率叫作数学准确率或教学质量得分。

## 手动体验

```powershell
python -m math_tutor chat --model models/qwen-0.5b --adapter runs/multiturn-sft-v1 --system-prompt-file data/versions/multiturn-v1/system_prompt.txt --keep-history --max-new-tokens 192
```

依次发送：

1. “我叫小林，喜欢画画，今天学数学有点累。”
2. “3盒铅笔，每盒8支，一共有多少支？只给提示，先别给答案。”
3. “我觉得是3加8，对吗？”
4. “明白了。现在每盒改成6支，盒数不变呢？”
5. “我刚才说喜欢什么？”

输入 /reset 后再发送条件不足的题：“几盒铅笔，每盒8支，共多少支？”观察是否先询问盒数。再补充“5盒”，检查能否结合前文解题。用 /exit 退出。

重点记录：有没有提前泄露答案、能否具体纠正加法错误、修改条件后是否沿用旧答案、闲聊是否自然、澄清后能否接着计算。一个例子成功不代表稳定泛化；要看完整逐轮记录和失败例。

## 本轮冻结数据

训练510段：单轮255、多轮数学153、通用多轮102，共1020个assistant回答；验证80段，共160个assistant回答。最大样本402 token。数学多轮完成独立请求的上下文状态复核；训练与验证family无交叉。只使用qwen3.8-flash，没有触发备用模型。

## 重新构建的执行顺序

生成器只写候选，审核器写审核结论，export_multiturn才生成训练文件与ready_for_training标志。当前版本已经冻结，生成器会拒绝覆盖。需要重新生成时，请先采用新的版本目录并统一修改各入口的版本路径。

```powershell
# 先通过进程环境提供 DASHSCOPE_API_KEY，勿把密钥写到脚本
python -m math_tutor.data.multiturn_data --workers 4 --budget 1400000
python -m math_tutor.data.audit_multiturn
python -m math_tutor.data.export_multiturn
python -m math_tutor.training.multiturn_sft --config configs/multiturn-sft-v1.json
```

## 本轮训练与测试结果

训练510段、验证80段，1 epoch，64次更新；训练273.47秒，train_loss=1.96965，eval_loss=1.82222。48个LoRA B张量非零，峰值CUDA已分配1.650GiB。Adapter与完成标志位于runs/multiturn-sft-v1。

| 规则检查（每项18轮） | 底座 | 旧SFT | 新多轮SFT |
|---|---:|---:|---:|
| 条件修改 | 12/18 | 13/18 | 11/18 |
| 提示约束与后续作答 | 2/18 | 3/18 | 14/18 |
| 纠错关键词 | 12/18 | 11/18 | 8/18 |
| 闲聊格式与爱好关键词 | 18/18 | 18/18 | 18/18 |

24段对话全部规则通过：底座6段、旧SFT9段、新SFT8段。它不是整体教学成功率。

解释：提示约束相关规则改善，但条件修改与纠错规则退化。原文中仍有允许解答后重复提示、修改每盒数量时擅自改变盒数、附和错误并编造未装满数量等失败。闲聊关键词命中不证明回答自然或严格遵循“只回答爱好”；例如“你喜欢画画吗？”仍会命中关键词。

本轮未达到全面改善多轮体验的结论，不作为现有最佳模型的替换依据。改进应在独立验证集上分析状态追踪、拒绝迎合和提示到解答的切换，另建未触碰回归集后再开展下一轮，避免依据这24段测试直接调参。

逐轮输出为results/multiturn-base-v1.jsonl、results/multiturn-old-sft-v1.jsonl、results/multiturn-sft-v1.jsonl；并列结果为reports/change-review-20261120/multiturn-comparison.json。
