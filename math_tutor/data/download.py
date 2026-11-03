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


def download_data(_):
    target = ROOT / "data/raw"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("math23k_train.json", "math23k_test.json"):
        path = target / name
        if not path.exists():
            req = urllib.request.Request(DATA_BASE + name, headers={"User-Agent": "math-tutor-starter"})
            # 下载完整并解析成功后才写正式文件，避免中断文件被误当有效数据。
            print("下载", name, flush=True)
            with urllib.request.urlopen(req, timeout=180) as r:
                content = r.read()
            tmp = path.with_suffix(".download")
            tmp.write_bytes(content)
            read_records(tmp)
            tmp.replace(path)
        rows = read_records(path)
        manifest[name] = {"url": DATA_BASE + name, "records": len(rows),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, "记录数", len(rows), flush=True)
    dump(target / "manifest.json", manifest)


def download_model(args):
    from huggingface_hub import snapshot_download
    dest = ROOT / args.model
    snapshot_download(repo_id=args.repo_id, local_dir=str(dest),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    dump(dest / "download_source.json", {"repo_id": args.repo_id})
    print("模型目录:", dest)
