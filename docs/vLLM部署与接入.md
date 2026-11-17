# vLLM 固定部署与项目接入

部署位置：WSL2 Ubuntu + Docker Desktop；模型只读挂载自 Windows E 盘。业务服务可继续在 Windows 的 math 环境运行，不加载第二份模型。

## 固定部署

在 Ubuntu 执行：

```bash
cd /mnt/e/desktop/LLM_Project/math-tutor
python3 deploy/vllm/freeze.py
```

脚本只使用已下载的镜像，记录真实 RepoDigest、image ID、GPU/驱动、容器内 PyTorch/CUDA/vLLM 版本、模型各文件 SHA256、Compose 配置哈希。
生成 deploy/vllm/deployment-lock.json、model-manifest.json、.env；后者为本机路径设置，Git 忽略。锁已存在时拒绝覆盖，明确更新才使用 --refresh。不下载 latest，不安装依赖，不修改模型。

提交版本锁和模型 manifest，保留 .env 在本机。要验证模型未改变，可重新计算 model-manifest 并对照旧版本；哈希本身不验证训练来源或质量。

已有手动运行的容器占用 8001 时，先在原前台终端 Ctrl+C 停止它。不要同时加载两份模型。

```bash
bash deploy/vllm/start.sh
docker compose --env-file deploy/vllm/.env -f deploy/vllm/compose.yaml logs -f
curl http://localhost:8001/v1/models
```

start.sh 在首次缺少 .env 时自动调用 freeze.py；每次启动都会验证模型文件、Compose 配置及镜像设置与锁一致，防止无意修改实验条件。Compose 使用镜像摘要及 pull_policy: never；保留 1024 上下文、单活跃序列、FP16、0.70 显存比例、eager 起始配置，挂载模型只读，并保留编译缓存卷。停止：bash deploy/vllm/stop.sh。停止不会删除镜像、模型或缓存卷。

## 启动项目网页（Windows Anaconda Prompt）

```bat
conda activate math
cd /d E:\desktop\LLM_Project\math-tutor
python -m pip install -r requirements-serve.txt
python -m math_tutor serve --backend vllm --base-url http://127.0.0.1:8001/v1 --remote-model math-sft --max-new-tokens 256 --port 8000
```

打开 http://127.0.0.1:8000 。业务 API 8000 调用推理 API 8001；base-url 必须包含 /v1。若业务服务也放入另一个容器，localhost 指向该容器，需要改用可达的推理服务地址。

GET /health 返回后端配置，remote 显存不伪造本地测量；GET /ready 实际访问上游 /v1/models 并确认配置模型存在，失败返回 503。

```bash
curl http://localhost:8000/ready
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"math-tutor","mode":"solve","messages":[{"role":"user","content":"小明有3个苹果，又买5个，一共多少个？"}],"temperature":0,"max_tokens":128}'
curl -N http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"math-tutor","mode":"solve","messages":[{"role":"user","content":"计算3+5"}],"stream":true,"temperature":0,"max_tokens":128}'
```

## 行为与边界

- --backend transformers 为原默认，原用法不变。vLLM 模式使用 httpx 异步调用，无生成锁，不加载本地模型/tokenizer。系统代理不会接管本机推理请求；可通过 MATH_TUTOR_API_KEY 环境变量提供上游鉴权。
- 沿用自动路由、规则答复与 finish_solution 计算器；自动解题仍先等待完整列式再计算，最终结果一次返回。显式 solve/chat 等生成模式支持逐段 SSE。
- 合并 SFT 服务只有 math-sft：默认所有模式均用此模型，与原底座/Adapter 分流行为不同。已支持 --remote-chat-model 指向同一上游提供的底座名称；首次不额外部署底座或动态 LoRA。--adapter 不接受本地路径，需在上游先配置 LoRA 并用模型名选择。
- 不使用共享可变状态保存请求结果；取消异步生成会关闭 HTTP 流。上游 timeout 为 504，连接/协议错误为 502，部分参数错误为 422，限流为 429；SSE 已开始后的错误通过 error 事件返回，客户端应按失败处理。
- remote 响应暂不包含 usage。辅导后处理会改变输出，未加载 tokenizer 时不伪造 token 统计；本地 Transformers 仍保留原统计。性能压测请直接读取推理服务的 usage 和指标。
- 上游负责 token 长度校验，不会静默截断；上下文 1024 是输入与输出合计，历史长时应清空历史或减少输出长度。

## 验证状态

代码测试可通过本机模拟上游验证，不代表真实 GPU 服务测试。助手执行环境访问 Docker 管道被拒绝，127.0.0.1:8001 未连通，因此真实 RepoDigest/版本锁需由 WSL 的 freeze.py 在可访问 Docker 的终端捕获；不要把 latest 或用户口述版本写成已核验摘要。
