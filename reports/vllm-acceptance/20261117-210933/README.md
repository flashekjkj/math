# vLLM 后续验收执行记录

日期：2026-11-17。项目：E:\desktop\LLM_Project\math-tutor。

## 已完成

- 新增 math_tutor/serve/acceptance.py：固定验证集100题、逐题数值评分、流式性能测量、跨后端协议一致性检查与逐题对比。
- 新增 deploy/vllm/finish_all.py：按阶段启动/停止本地模型、通过WSL启动固定版本容器、接入远程网关、生成结果并保留最终网页服务。支持已完成基线续跑，不覆盖中断或旧实验。
- 实际完成 Windows math 环境 + RTX3050 的 Transformers 合并模型服务、普通/流式接口、100题验证与36次流式性能请求；完成后已停止本次创建的本地服务释放显存。
- 验证集数值正确率18/100=18%，请求失败0，解析率99%；与已有验证集汇总一致。
- 输入长度76–131 token，输出上限192，本次题目均可容纳于vLLM的1024总上下文预算。
- 完整代码测试91项通过。新增5项测量/比较测试，覆盖请求失败计入分母、提前断流、正常SSE、协议不一致拒绝比较等。
- 已更新 docs/vLLM验收与性能对照.md 与 README，保留真实数据和环境版本。

## 本机真实测量

每档12次请求。经项目HTTP网关测量；输出token为同一tokenizer重新分词，不是GPU引擎内部计数。

| 并发 | 失败 | 端到端延迟p50(s) | p95(s) | 输出token/s | 首段文本p50(s) |
|---|---|---|---|---|---|
| 1 | 0 | 1.66 | 2.19 | 10.81 | 0.091 |
| 2 | 0 | 3.80 | 4.38 | 9.78 | 1.861 |
| 4 | 0 | 6.29 | 7.15 | 11.09 | 4.832 |

当前本地后端使用生成锁，并发增加主要引入排队。该小规模结果不是生产容量，不与历史不同口径31 tokens/s混算速度提升。首段文本不等于严格首token，不提供虚构TPOT。数值成绩不证明列式或教学正确性。

## 尚未完成的运行任务

Docker Desktop命名管道拒绝访问，WSL返回E_ACCESSDENIED；本机8001未监听。权限申请接口无法表示Docker命名管道路径，普通项目文件权限不能解除此限制。未生成真实镜像摘要、未启动vLLM容器、未取得vLLM性能/效果结果，也未声称完成双后端对照。

请在自己的Windows Anaconda Prompt执行：

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python deploy/vllm/finish_all.py --resume reports/vllm-acceptance/20261117-210933
```

如默认WSL发行版不是Ubuntu，添加 --distro 实际发行版名。执行前停止已有GPU容器或训练；脚本不会擅自停止其他任务。Docker Desktop应已启动。

完成后自动生成 comparison.json、运行时版本锁副本和结果README；远程网页保留在 http://127.0.0.1:8000 。若失败，run-status.json 与网关日志保存具体原因。

合并SFT部署仍默认所有模式使用同一模型；底座/Adapter分流需要上游提供独立模型名，可通过已有 --remote-chat-model 配置。未擅自改用动态LoRA，以免改变此次合并模型对照条件。