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


def load_inference(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise SystemExit("本套推理命令需要 CUDA GPU。先执行 python starter.py check")
    path = str((ROOT / args.model).resolve())
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16,
                                               device_map={"": 0}, trust_remote_code=False)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str((ROOT / args.adapter).resolve()))
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, messages, max_new_tokens):
    import torch
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    # 保护入门配置，长历史请重新开始对话；不静默截断。
    if inputs["input_ids"].shape[1] > 2048:
        raise ValueError("输入超过2048 token，请缩短问题或输入 /reset")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def chat(args):
    from math_tutor.inference.calculator import TOOL_SYSTEM, execute
    tokenizer, model = load_inference(args)
    tool_mode = getattr(args, 'tool_calculation', False)
    calculator_only = getattr(args, 'calculator_only', False)
    if tool_mode and calculator_only:
        raise ValueError('Choose one calculator mode')
    messages = [{"role": "system", "content": TOOL_SYSTEM if tool_mode else get_system_prompt(args)}]
    print("输入 /exit 退出，/reset 清空历史。当前模型只做解题入门实验。")
    while True:
        q = input("你：").strip()
        if q == "/exit":
            break
        if q == "/reset":
            messages = messages[:1]
            continue
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        try:
            answer = generate(tokenizer, model, messages, args.max_new_tokens)
        except ValueError as e:
            messages.pop()
            print(e)
            continue
        if tool_mode or calculator_only:
            result = execute(answer, require_call=tool_mode)
            print("模型与计算器：", result['response'] if result['ok'] else '计算失败：'+result['error']+'\n原始输出：'+answer)
        else:
            print("模型：", answer)
        messages.append({"role": "assistant", "content": answer})
