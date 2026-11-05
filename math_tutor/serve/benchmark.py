"""推理基准：加载耗时、显存占用、首字延迟、生成速度。

同一提示跑多次取均值，输出 JSON 报告到 results/，不覆盖旧文件。
示例：
python -m math_tutor benchmark --model models/math-sft-0.5b --runs 3
python -m math_tutor benchmark --model models/qwen-0.5b --adapter runs/local-sft-v1 --output results/benchmark-adapter.json
"""
import argparse
import json
import time

from math_tutor.common import ROOT, dump
from .engine import ModelEngine

DEFAULT_PROMPT = '工厂生产了1015筒毛球，每筒6个。一共生产了多少个羽毛球？'


def benchmark(args):
    dest = ROOT / args.output
    if dest.exists():
        raise SystemExit('输出文件已存在，请换一个 --output；不覆盖旧结果')
    import torch
    t0 = time.time()
    engine = ModelEngine(args.model, args.adapter, args.max_new_tokens)
    load_seconds = time.time() - t0
    messages = [dict(role='user', content=args.prompt)]

    # 预热一次，避免首次 CUDA 初始化计入正式成绩
    t0 = time.time()
    engine.generate(messages, args.max_new_tokens, temperature=0)
    warmup_seconds = time.time() - t0

    runs = []
    for i in range(args.runs):
        t0 = time.time()
        first_token_seconds = None
        pieces = []
        for piece in engine.stream(messages, args.max_new_tokens, temperature=0):
            if first_token_seconds is None and piece:
                first_token_seconds = time.time() - t0
            pieces.append(piece)
        elapsed = time.time() - t0
        text = ''.join(pieces)
        completion_tokens = len(engine.tokenizer.encode(text, add_special_tokens=False))
        gen_seconds = max(elapsed - (first_token_seconds or 0), 1e-6)
        runs.append(dict(run=i + 1, total_seconds=round(elapsed, 2),
                         first_token_seconds=round(first_token_seconds or 0, 2),
                         completion_tokens=completion_tokens,
                         tokens_per_second=round(completion_tokens / gen_seconds, 1)))
    avg = lambda key: round(sum(r[key] for r in runs) / len(runs), 2) if runs else None
    report = dict(model=args.model, adapter=args.adapter, device=engine.device,
                  prompt=args.prompt,
                  load_seconds=round(load_seconds, 2), warmup_seconds=round(warmup_seconds, 2),
                  vram_allocated_gb=engine.info()['vram_allocated_gb'],
                  vram_total_gb=engine.info()['vram_total_gb'],
                  runs=runs,
                  averages=dict(first_token_seconds=avg('first_token_seconds'),
                                total_seconds=avg('total_seconds'),
                                completion_tokens=avg('completion_tokens'),
                                tokens_per_second=avg('tokens_per_second')),
                  note='贪心生成（temperature=0）；first_token_seconds 为纯生成部分的首字延迟，'
                       '不含请求与分词开销。')
    dump(dest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
