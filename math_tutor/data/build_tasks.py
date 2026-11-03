"""Validate and export manually reviewed solve/diagnosis/tutor records; never synthesize labels."""
import argparse
import hashlib
from pathlib import Path
from math_tutor.common import ROOT, dump, write_lines, read_records, SYSTEM
from math_tutor.data.prepare import family

def build(args):
    from transformers import AutoTokenizer
    source = ROOT / args.input
    out = ROOT / args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit('输出目录非空，请使用新的数据版本名')
    tok = AutoTokenizer.from_pretrained(str(ROOT / args.model), local_files_only=True)
    frozen = ROOT / 'data/processed/test.jsonl'
    forbidden = {r['family_id'] for r in read_records(frozen)} if frozen.exists() else set()
    families, ids, data, rejects = {}, set(), {'train': [], 'valid': []}, []
    for index, row in enumerate(read_records(source)):
        try:
            split = row['split']
            if split not in data: raise ValueError('split必须为train或valid；测试情境单独保管')
            if row['id'] in ids: raise ValueError('重复id')
            ids.add(row['id'])
            q = row['question']
            fid = row.get('family_id') or family(q)
            if fid in forbidden or family(q) in forbidden: raise ValueError('与冻结测试题家族重叠')
            if fid in families and families[fid] != split: raise ValueError('同家族跨train/valid')
            families[fid] = split
            if row.get('reviewed') is not True: raise ValueError('没有人工审核确认 reviewed=true')
            conv = row['conversations']
            roles = [m['from'] for m in conv]
            if roles[0] == 'system': roles = roles[1:]
            if not roles or roles[-1] != 'gpt' or any(role != ('human' if i % 2 == 0 else 'gpt') for i,role in enumerate(roles)):
                raise ValueError('对话应human/gpt交替，最后为gpt')
            mapping = {'system':'system','human':'user','gpt':'assistant'}
            messages = [{'role': mapping[m['from']], 'content': m['value']} for m in conv]
            if len(tok.apply_chat_template(messages, tokenize=True, return_dict=False)) > args.max_length:
                raise ValueError('完整对话超过长度限制')
            data[split].append({'conversations': conv})
        except (KeyError, ValueError, IndexError, TypeError) as e:
            rejects.append({'index': index, 'id': row.get('id'), 'reason': str(e)})
    # Validation is transactional: invalid inputs do not produce a partial trainable dataset.
    if rejects:
        dump(out / 'validation_errors.json', rejects)
        raise SystemExit(f'{len(rejects)}条记录不通过，请修订输入并使用新的输出目录；未导出训练文件')
    if not data['train'] or not data['valid']:
        raise SystemExit('训练集与验证集都需要至少一条记录')
    for split, rows in data.items(): write_lines(out / f'sft_{split}/{split}.jsonl', rows)
    dump(out / 'manifest.json', {'input': str(source), 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
         'counts': {k:len(v) for k,v in data.items()}, 'max_length':args.max_length,
         'note':'reviewed是人工声明，程序仅检查结构、长度和启发式隔离，不能自动认证教学质量。'})
    print('已导出', {k:len(v) for k,v in data.items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='models/qwen-0.5b')
    p.add_argument('--max-length', type=int, default=512)
    build(p.parse_args())
if __name__=='__main__': main()
