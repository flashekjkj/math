from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = "你是中文数学学习助手。请根据题意列出计算式并计算，不要编造条件。最后一行以“答案：”开头，后面填写计算出的数字，例如“答案：192”。不要照抄“数值”两个字。"
DATA_BASE = "https://raw.githubusercontent.com/SCNU203/Math23k/main/"
def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_records(path):
    """兼容 JSON 数组、JSONL，以及 Math23K 连续的多行 JSON 对象。"""
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("["):
        return json.loads(text)
    decoder = json.JSONDecoder()
    rows, pos = [], 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        row, pos = decoder.raw_decode(text, pos)
        if not isinstance(row, dict):
            raise ValueError("每条记录必须是对象")
        rows.append(row)
    return rows


def get_system_prompt(args):
    path = getattr(args, "system_prompt_file", None)
    return (ROOT / path).read_text(encoding="utf-8-sig").strip() if path else SYSTEM
