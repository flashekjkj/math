import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import operator
from pathlib import Path
import random
import re
import subprocess
import sys
import urllib.request
from math_tutor.common import ROOT, MODEL_ID, SYSTEM, DATA_BASE, dump, write_lines, read_records, get_system_prompt
from .scoring import extract_details, score_response, summarize, VERSION as SCORER_VERSION
from math_tutor.inference.runtime import load_inference, generate


def extract_answer(text):
    # 保留旧函数接口，评分细节由 math_scoring.py 负责。
    return extract_details(text)["prediction"]


def rescore(args):
    source = ROOT / args.input
    dest = ROOT / args.output
    if dest.exists() or dest.with_suffix(".summary.json").exists():
        raise SystemExit("输出文件已存在，请指定新的 --output；不会覆盖原始评测")
    rows = read_records(source)
    outputs = []
    for row in rows:
        result = dict(row)
        result["previous_prediction"] = row.get("prediction")
        result["previous_correct"] = row.get("correct")
        result.update(score_response(row["question"], row["response"], row["reference"]))
        outputs.append(result)
    write_lines(dest, outputs)
    original_summary = source.with_suffix(".summary.json")
    summary = json.loads(original_summary.read_text(encoding="utf-8-sig")) if original_summary.exists() else {}
    summary.update(summarize(outputs))
    summary.update(rescored_from=str(source),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   generation_unchanged=True,
                   system_prompt_note="复用旧回答，保留旧提示生成条件；未用新提示重新生成")
    dump(dest.with_suffix(".summary.json"), summary)
    review = [r for r in outputs if r["status"] in ("unparsed", "conflict", "needs_review")]
    write_lines(dest.with_suffix(".review.jsonl"), review)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate(args):
    dest = ROOT / args.output
    if dest.exists():
        raise SystemExit("评测输出已存在，请指定新的 --output")
    rows = read_records(ROOT / args.data)
    if args.limit:
        rows = rows[:args.limit]
    tokenizer, model = load_inference(args)
    outputs = []
    for i, row in enumerate(rows):
        messages = [{"role": "system", "content": get_system_prompt(args)}, {"role": "user", "content": row["question"]}]
        answer = generate(tokenizer, model, messages, args.max_new_tokens)
        score = score_response(row["question"], answer, row["answer"])
        outputs.append({"id": row["id"], "question": row["question"], "reference": row["answer"],
                        "response": answer, **score})
        print(f"{i+1}/{len(rows)} status={score['status']}", flush=True)
    write_lines(dest, outputs)
    parsed = sum(r["prediction"] is not None for r in outputs)
    count = len(outputs)
    summary = {"model": args.model, "adapter": args.adapter, "data": args.data,
               "data_sha256": hashlib.sha256((ROOT / args.data).read_bytes()).hexdigest(),
               "max_new_tokens": args.max_new_tokens, "do_sample": False, "n": count,
               "numeric_accuracy": sum(r["correct"] is True for r in outputs) / count if count else 0,
               "parse_rate": parsed / count if count else 0,
               "note": "这是带答案格式要求的数值正确率；需人工复核解析失败、单位与解法，不能代表教学效果。"}
    summary.update(summarize(outputs))
    summary["system_prompt"] = get_system_prompt(args)
    dump(dest.with_suffix(".summary.json"), summary)
    write_lines(dest.with_suffix(".review.jsonl"),
                [r for r in outputs if r["status"] in ("unparsed", "conflict", "needs_review")])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
