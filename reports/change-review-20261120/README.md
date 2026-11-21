# 最后一次 Git 提交后的新增工作（本次修改前）

比较基准：868ce6a，2026-11-17 21:23:53 +0800，提交说明 vLLM。

1. vLLM完整验收与2026-11-19性能补测：加入vLLM逐题输出、协议/健康/性能记录、双后端比较、metrics探针和GPU采样。并发2的网关输出吞吐从9.78到56.03 token/s；每档12请求，属于当前栈小规模测量。
2. 部署可复现性：加入deployment-lock.json；freeze.py支持从已有可信锁重新生成本机挂载配置；finish_all.py支持直接使用Windows Docker CLI启动，也可显式经WSL；start/stop脚本有换行规范调整。
3. 历史证据恢复：reports/sft-history-recovered保存扩数、训练轮次、1.5B QLoRA和外部API记录，附文件哈希和题目身份核对。找回记录不代表本次重新训练或调用API。
4. 文档整理：README大幅更新，补充结果边界、部署、历史实验；新增项目文件导览和vLLM补测报告；更新验收文档。
5. 工程规范与测试：新增.editorconfig、.gitattributes、tests/test_deployment.py；tests/test.py有原有修改。

以上是本次操作前已存在的未提交内容，不全部归为本次新增。Git历史没有改写，本次也没有提交或推送。训练权重和部分数据受.gitignore影响，不能只凭git diff判断磁盘上是否存在。
