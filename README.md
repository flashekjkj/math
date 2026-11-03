# Math Tutor：中文数学模型微调项目

独立项目代码位于math_tutor/，不依赖同级MedicalGPT目录。当前完成0.5B数学解题LoRA，正在向错因诊断与多轮辅导扩展。

- [第一轮实验评估](docs/实验评估.md)
- [文件结构与学习顺序](docs/项目结构.md)
- [后续训练逐步命令](docs/后续训练步骤.md)
- 上游来源与许可：third_party/provenance.json、third_party/MedicalGPT-LICENSE
- 模型、训练检查点、数据集和本地备份不纳入 Git；数据与模型通过项目命令下载、生成。

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m math_tutor --help
python -m unittest discover -s tests -v
python -m math_tutor chat --adapter runs/local-sft-v1
```

训练：python -m math_tutor train --config configs/solve-v2-3k.json
正式训练前按后续文档生成对应版本数据。不要覆盖旧实验目录。
旧python starter.py命令保留兼容，主体代码已模块化。
首次安装：配置匹配GPU的PyTorch，再安装requirements.txt；高级算法按需安装requirements-advanced.txt。
当前环境完整版本记录在requirements.lock.txt。新服务器请先验证版本与GPU，不假定Windows环境可直接搬迁。

已发现当前评分只检查数值而非完整推导；第一轮48条数值判对中4条算式不一致，后续需分项评测。详情见实验评估。

