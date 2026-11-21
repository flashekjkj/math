# vLLM 验收与性能对照

## 一次执行后续流程

使用 Windows Anaconda Prompt 的 math 环境：

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python deploy/vllm/finish_all.py
```

默认通过 Windows Docker CLI 直接控制 Docker Desktop 的 WSL2 引擎，不再调用 Ubuntu Shell。只有显式添加 --via-wsl 时才通过 Ubuntu；此时可用 --distro 指定发行版。本脚本不会删除模型或训练产物，不会安装或下载镜像。

执行前停止正在运行的 GPU 容器及训练任务，释放 4GB 显存；脚本检测到已有 GPU 容器会停止流程，不擅自终止其他任务。端口 8000、8002 已被占用时也会明确报错。

执行顺序：

1. 检查 Docker 控制通路。
2. 启动 Transformers 合并 SFT 服务（8002），检查接口、跑验证集100题，再对相同12条请求测并发1/2/4；停止该进程释放显存。
3. 使用 Windows Docker CLI 启动锁定摘要的 Compose vLLM 服务（8001），启动项目远程网关（8000）。
4. 在相同协议下检查普通/流式接口、验证集与性能，生成逐题对照和汇总。
5. 成功时保留 vLLM 容器与远程网关，网页地址 http://127.0.0.1:8000 。

仅基线：python deploy/vllm/finish_all.py --baseline-only 。已完成基线的目录可以通过 --resume reports/vllm-acceptance/<运行目录> 续跑，脚本会检查评测协议一致性；中断的不完整阶段拒绝覆盖，使用新目录重跑。

## 输出

reports/vllm-acceptance/<时间戳>/：

- run-status.json：当前阶段、错误或完成标记；不能只凭目录存在宣称成功。
- gateway-*.log：服务启动日志。
- transformers/ 与 vllm/：protocol.json、predictions.jsonl、evaluation.json、performance.json、ready/smoke/warmup/health、completion.json。
- comparison.json：相同题目ID与协议的正确/退化/回答变化数量、性能原始结果。
- runtime-lock-copy.json：本次实际容器镜像和软件环境记录。
- README.md：真实完成后的简表。

## 结果解释

使用验证集，不再消耗测试集。数值评分器 numeric-v2，不能替代列式/单位/推导与教学质量评测。失败请求保留在准确率分母；HTTP失败、断流或缺失DONE不计为成功。

两后端均从项目HTTP网关测量，以显式solve模式和相同system提示生成，不调用自动辅导参考答案。temperature=0、top_p=1、输出上限192，题目ID与数据哈希固定。同一权重不保证跨内核逐字一致。

测量端到端 p50/p95 延迟、首段非空文本 p50/p95、成功请求吞吐、重新分词后的输出token吞吐及失败率。首段文本不等于严格首token，不虚构 TPOT；输出token由同一模型tokenizer在客户端重新分词，排除EOS，与上游usage可能不同。一次小规模运行不代表生产容量或统计显著性。

当前vLLM部署限制单活跃序列；客户端并发2/4主要测排队和服务行为，不能直接宣称验证了大批处理收益。后续调整 max-num-seqs、前缀缓存等参数需创建新版本锁与新实验。

历史 benchmark 的约31 tokens/s为不同测量口径，不用于本次速度提升百分比。

## 当前执行限制

助手执行环境的 WSL/Docker 命名管道访问被拒绝，不能直接申请可表示的文件路径解除。请在自己的普通 Windows Anaconda Prompt 运行上述命令。不要关闭系统安全设置、开放未鉴权Docker TCP端口或重新安装驱动来绕过。

## 2026-11-19 修复

start.sh/stop.sh 已转换为 UTF-8 无 BOM 的 LF 换行，增加 .gitattributes 与 .editorconfig 防止重新变成 CRLF。默认 Windows 编排流程直接调用 freeze.py 和 docker compose，绕开 Shell 换行差异。已有版本锁可使用 --configure-local 重新生成本机挂载路径，不刷新镜像摘要，且仍验证模型与配置一致。

请在 Windows math 环境续跑：

```powershell
python deploy/vllm/finish_all.py --resume reports/vllm-acceptance/20261117-210933
```

已完成的 Transformers 100题与性能基线不重跑。此命令仍需你的终端能正常执行 docker info。
