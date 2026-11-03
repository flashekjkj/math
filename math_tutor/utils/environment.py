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


def check(_):
    print("Python:", sys.version, "\n解释器:", sys.executable)
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes"):
        try:
            print(name, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name, "未安装")
    import torch
    print("CUDA 可用:", torch.cuda.is_available(), "PyTorch CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("显存 GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        print("BF16:", torch.cuda.is_bf16_supported())
        x = torch.ones(8, device="cuda")
        print("CUDA 运算检查:", x.sum().item())
