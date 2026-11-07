# Math Tutor：中文数学模型微调项目

独立项目代码位于math_tutor/，不依赖同级MedicalGPT目录。当前完成0.5B数学解题LoRA，正在向错因诊断与多轮辅导扩展。

- [当前训练与工具计算使用说明](docs/使用说明.md)
- [部署与网页演示](docs/部署使用.md)
- 部署依赖：requirements-serve.txt（fastapi、uvicorn，在训练环境上附加安装）
- 上游来源与许可：third_party/provenance.json、third_party/MedicalGPT-LICENSE
- 模型、训练检查点、数据集和本地备份不纳入 Git；数据与模型通过项目命令下载、生成。

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m math_tutor --help
python -m unittest discover -s tests -v
python -m math_tutor chat --model models/math-relation-tool-0.5b --tool-calculation --max-new-tokens 256
python -m math_tutor serve --model models/math-sft-0.5b
python -m math_tutor benchmark --model models/math-sft-0.5b
```

当前训练：python -m math_tutor train --config configs/relation-tool.json
训练数据为原1000题增加算式结构关系说明与计算器调用；不扩数据，不训练模型执行算术。配置和数据已生成，不要覆盖旧实验目录。
旧python starter.py命令保留兼容，主体代码已模块化。
首次安装：配置匹配GPU的PyTorch，再安装requirements.txt；高级算法按需安装requirements-advanced.txt。
当前环境完整版本记录在requirements.lock.txt。新服务器请先验证版本与GPU，不假定Windows环境可直接搬迁。

当前评分只检查数值而非完整推导；工具执行正确不代表列式符合题意。旧SFT原始成绩和工具执行成绩分别保存，新实验汇总见 results/relation-tool-comparison.json。


GRPO：参见 [训练命令与实现](docs/GRPO训练说明.md) 和 [第一轮实验结果](docs/GRPO实验结果.md)。
