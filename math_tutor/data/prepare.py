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
from math_tutor.common import ROOT, MODEL_ID, SYSTEM, DATA_BASE, dump, write_lines, read_records


def expression(text):
    """仅支持白名单内的四则、百分数、带分数和有限幂，绝不使用 eval。"""
    s = str(text).strip().replace(" ", "").replace("（", "(").replace("）", ")")
    s = s.replace("×", "*").replace("÷", "/").replace("：", "/")
    if "=" in s:
        left, s = s.split("=", 1)
        if left.lower() != "x":
            raise ValueError("不支持的等号左侧")
    s = re.sub(r"(\d+)\((\d+)/(\d+)\)", r"(\1+\2/\3)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)
    s = s.replace("^", "**")
    if len(s) > 300 or not re.fullmatch(r"[\d.()+*/\-]+", s):
        raise ValueError("不支持的表达式")
    tree = ast.parse(s, mode="eval")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}

    def visit(node, depth=0):
        if depth > 30:
            raise ValueError("表达式过深")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            v = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = visit(node.operand, depth + 1)
            v = -v if isinstance(node.op, ast.USub) else v
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(b) > 8 or abs(a) > 1e6):
                raise ValueError("幂过大")
            v = ops[type(node.op)](a, b)
        else:
            raise ValueError("不支持的语法")
        if isinstance(v, complex) or not math.isfinite(v) or abs(v) > 1e12:
            raise ValueError("结果超出范围")
        return v
    return s, visit(tree.body)


def family(question):
    # 启发式去重：去掉空白、常见标点，将阿拉伯数值替换为占位符。
    # 不能识别全部中文数字及语义改写，仍需人工复核近重复。
    q = re.sub(r"\d+(?:\.\d+)?", "<N>", question)
    q = re.sub(r"[\s，。！？,.!?＝=、；;：:]", "", q)
    return hashlib.sha256(q.encode()).hexdigest()


def convert(rows, split):
    clean, rejected = [], []
    for index, row in enumerate(rows):
        try:
            q = str(row.get("original_text", "")).strip()
            if not q:
                raise ValueError("缺题目")
            eq, value = expression(row["equation"])
            _, answer = expression(row["ans"])
            if not math.isclose(value, answer, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("方程与答案不一致")
            result = format(answer, ".10g")
            target = f"列式：{eq}\n答案：{result}"
            # 保留元数据用于审计，但最终训练文件仅输出 conversations。
            clean.append({"id": f"{split}:{row.get('id', index)}", "family_id": family(q),
                          "question": q, "equation": eq, "answer": answer,
                          "source": "Math23K / SCNU203 mirror", "task": "solve",
                          "conversations": [{"from": "system", "value": SYSTEM},
                                            {"from": "human", "value": q},
                                            {"from": "gpt", "value": target}]})
        except (KeyError, ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError) as e:
            rejected.append({"source_split": split, "index": index,
                             "id": row.get("id"), "reason": str(e)})
    return clean, rejected


def prepare(args):
    raw = ROOT / "data/raw"
    output_root = ROOT / getattr(args, "output_root", "data")
    train, reject_a = convert(read_records(raw / "math23k_train.json"), "train")
    test, reject_b = convert(read_records(raw / "math23k_test.json"), "test")
    # 官方测试文件全部保留在测试侧；所有同 family 的训练记录从训练池排除。
    seen, unique_test = set(), []
    for row in test:
        if row["family_id"] not in seen:
            unique_test.append(row)
            seen.add(row["family_id"])
    train_pool, duplicates = [], 0
    for row in train:
        if row["family_id"] in seen:
            duplicates += 1
            continue
        seen.add(row["family_id"])
        train_pool.append(row)
    valid = [r for r in train_pool if int(r["family_id"][:8], 16) % 10 == 0]
    train_pool = [r for r in train_pool if int(r["family_id"][:8], 16) % 10 != 0]
    rng = random.Random(42)
    for rows in (train_pool, valid, unique_test):
        rng.shuffle(rows)
    splits = {"train": train_pool[:args.train_size], "valid": valid[:args.valid_size],
              "test": unique_test[:args.test_size]}
    if any(not rows for rows in splits.values()):
        raise ValueError("有集合为空，请检查原始文件与清洗报告")
    for split, rows in splits.items():
        write_lines(output_root / f"processed/{split}.jsonl", rows)
        if split != "test":
            write_lines(output_root / f"sft_{split}/{split}.jsonl",
                        [{"conversations": r["conversations"]} for r in rows])
    write_lines(output_root / "rejected.jsonl", reject_a + reject_b)
    report = {"clean_source_train": len(train), "clean_source_test": len(test),
              "rejected": len(reject_a) + len(reject_b), "duplicate_families_removed": duplicates,
              "seed": 42, "splits": {k: len(v) for k, v in splits.items()},
              "warning": "仅验证表达式与答案一致；没有证明表达式符合题意。需人工抽检。"}
    dump(output_root / "quality_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.model), trust_remote_code=False)
    reports = {}
    data_root = ROOT / getattr(args, "data_root", "data")
    mapping = {"system": "system", "human": "user", "gpt": "assistant"}
    for split in ("train", "valid"):
        rows = read_records(data_root / f"processed/{split}.jsonl")
        kept, lengths = [], []
        for row in rows:
            messages = [{"role": mapping[m["from"]], "content": m["value"]}
                        for m in row["conversations"]]
            size = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False))
            lengths.append(size)
            if size <= args.max_length:
                kept.append({"conversations": row["conversations"]})
        if not kept:
            raise SystemExit(f"{split} 没有样本满足长度要求")
        write_lines(data_root / f"sft_{split}/{split}.jsonl", kept)
        reports[split] = {"before": len(rows), "after": len(kept), "longest": max(lengths),
                          "p95": sorted(lengths)[min(len(lengths)-1, int(len(lengths)*0.95))]}
    reports.update(model=args.model, max_length=args.max_length)
    dump(data_root / "length_report.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print("只导出完整且不超长的训练/验证样本；未修改测试集。")
    print("这不是上游训练标签检查；首次训练仍需查看上游预处理日志与有效样本数。")
