import argparse
from math_tutor.common import *
from math_tutor.data.download import download_data, download_model
from math_tutor.data.prepare import prepare, audit
from math_tutor.utils.environment import check
from math_tutor.training.launcher import launch
from math_tutor.inference.runtime import chat
from math_tutor.evaluation.runner import evaluate, rescore
from math_tutor.serve.api import serve
from math_tutor.serve.benchmark import benchmark

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", check), ("download-data", download_data)):
        sub.add_parser(name).set_defaults(func=fn)
    p = sub.add_parser("prepare-data")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--valid-size", type=int, default=100)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--output-root", default="data")
    p.set_defaults(func=prepare)
    p = sub.add_parser("download-model")
    p.add_argument("--repo-id", default=MODEL_ID)
    p.add_argument("--model", default="models/qwen-0.5b")
    p.set_defaults(func=download_model)
    p = sub.add_parser("audit")
    p.add_argument("--model", default="models/qwen-0.5b")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--data-root", default="data")
    p.set_defaults(func=audit)
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stage", choices=["sft","dpo","pt","orpo","grpo","rm","ppo","opd"], default="sft")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=launch)
    for command, fn in (("chat", chat), ("evaluate", evaluate)):
        p = sub.add_parser(command)
        p.add_argument("--model", default="models/qwen-0.5b")
        p.add_argument("--adapter")
        p.add_argument("--system-prompt-file")
        p.add_argument("--max-new-tokens", type=int, default=192)
        if command == "chat":
            p.add_argument("--tool-calculation", action="store_true")
            p.add_argument("--calculator-only", action="store_true", help="Execute old-model formulas without requiring JSON calls")
        if command == "evaluate":
            p.add_argument("--data", default="data/processed/test.jsonl")
            p.add_argument("--output", required=True)
            p.add_argument("--limit", type=int, default=0)
        p.set_defaults(func=fn)
    p = sub.add_parser("rescore", help="仅重新评分已有回答，不加载模型")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=rescore)
    p = sub.add_parser("serve", help="启动网页演示与 OpenAI 兼容推理服务")
    p.add_argument("--model", default="models/math-sft-0.5b")
    p.add_argument("--adapter")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.set_defaults(func=serve)
    p = sub.add_parser("benchmark", help="推理基准：加载耗时、显存、首字延迟、tokens/s")
    p.add_argument("--model", default="models/math-sft-0.5b")
    p.add_argument("--adapter")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--prompt", default="工厂生产了1015筒毛球，每筒6个。一共生产了多少个羽毛球？")
    p.add_argument("--output", default="results/benchmark-math-sft.json")
    p.set_defaults(func=benchmark)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
