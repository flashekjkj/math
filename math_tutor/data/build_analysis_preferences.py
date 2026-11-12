"""Reuse exact explanation SFT targets and mine executable, numerically wrong responses."""
import argparse
from collections import Counter
import json
import math
import random

from math_tutor.common import ROOT, read_records, write_lines
from math_tutor.data.preference_io import sha256, atomic_json, run_lock
from math_tutor.inference.calculator import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/versions/analysis-preference-v1')
    parser.add_argument('--train-size',type=int,default=200)
    parser.add_argument('--valid-size',type=int,default=30)
    args = parser.parse_args()
    directory = ROOT/args.output
    model_path = ROOT/'models/math-analysis-0.5b'
    source = ROOT/'data/versions/teacher-analysis-v1/accepted.jsonl'
    targets_path = ROOT/'data/versions/teacher-analysis-v1/sft_train/train.jsonl'
    targets = {json.dumps(r['conversations'],ensure_ascii=False,sort_keys=True)
               for r in read_records(targets_path)}
    references = {r['id']:r for r in read_records(ROOT/'data/processed/train.jsonl')}
    frozen = {r['family_id'] for split in ('valid','test')
              for r in read_records(ROOT/f'data/processed/{split}.jsonl')}
    disputed = set()
    earlier = ROOT/'data/versions/preference-sample-v1/teacher_reviews.jsonl'
    if earlier.exists():
        disputed = {r['id'] for r in read_records(earlier) if r['review']['reference_status'] != 'correct'}
    rows = []
    for row in read_records(source):
        if row['id'] in disputed:
            continue
        reference = references[row['id']]
        if reference['family_id'] in frozen or row['question'] != reference['question']:
            raise ValueError('Question identity or split mismatch')
        if json.dumps(row['conversations'],ensure_ascii=False,sort_keys=True) not in targets:
            raise ValueError('Chosen is not an exact SFT training target')
        chosen = row['conversations'][-1]['value']
        checked = execute(chosen,require_call=False)
        if not checked['ok'] or not math.isclose(checked['value'],reference['answer'],rel_tol=1e-5,abs_tol=1e-5):
            continue
        rows.append(row | {'answer':reference['answer']})
    random.Random(42).shuffle(rows)
    manifest = dict(model=str(model_path), model_sha256={p.name:sha256(p) for p in model_path.iterdir() if p.is_file()},
                    source_sha256=sha256(source),sft_targets_sha256=sha256(targets_path),
                    train_size=args.train_size,valid_size=args.valid_size,
                    excluded_disputed_ids=sorted(disputed),seed=42,max_new_tokens=256,
                    sampling='greedy; one original response per question',
                    preference_validation='held-out families from SFT training pool; original validation/test untouched')
    with run_lock(directory,'build'):
        path = directory/'manifest.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Source or model changed; select a new output version')
        atomic_json(path,manifest)
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.float16,
                    device_map={'':0},attn_implementation='sdpa').eval()
        eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos,int) else set(eos)
        pairs = {'train':[],'valid':[]}
        counts = Counter()
        for offset in range(0,len(rows),4):
            batch = rows[offset:offset+4]
            pending = [r for r in batch if not (directory/'samples'/f'{r["id"].replace(":","_")}.json').exists()]
            if pending:
                messages = [[{'role':{'system':'system','human':'user'}[m['from']],'content':m['value']}
                             for m in row['conversations'][:-1]] for row in pending]
                inputs = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                           padding=True,return_dict=True,return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs,max_new_tokens=256,do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id)
                for row,tail in zip(pending,generated[:,inputs.input_ids.shape[1]:]):
                    ids = tail.tolist()
                    length = next((i+1 for i,t in enumerate(ids) if t in eos_ids),len(ids))
                    text = tokenizer.decode(ids[:length],skip_special_tokens=True)
                    atomic_json(directory/'samples'/f'{row["id"].replace(":","_")}.json',
                        dict(id=row['id'],raw_response=text,finish_reason='eos' if ids[length-1] in eos_ids else 'length'))
            for row in batch:
                sample = json.loads((directory/'samples'/f'{row["id"].replace(":","_")}.json').read_text(encoding='utf-8'))
                counts['sampled'] += 1
                tool = execute(sample['raw_response'],require_call=False)
                if sample['finish_reason'] != 'eos' or not tool['ok']:
                    counts['incomplete_or_unexecutable'] += 1
                    continue
                if math.isclose(tool['value'],row['answer'],rel_tol=1e-5,abs_tol=1e-5):
                    counts['correct_response'] += 1
                    continue
                split = 'valid' if int(row['family_id'][:8],16)%5 == 0 else 'train'
                limit = args.valid_size if split == 'valid' else args.train_size
                if len(pairs[split]) >= limit:
                    continue
                prompt = tokenizer.apply_chat_template([
                    {'role':{'system':'system','human':'user'}[m['from']],'content':m['value']}
                    for m in row['conversations'][:-1]],tokenize=False,add_generation_prompt=True)
                chosen = row['conversations'][-1]['value']
                rejected = sample['raw_response']
                if any(len(tokenizer.encode(prompt+text,add_special_tokens=False))+1 > 512 for text in (chosen,rejected)):
                    counts['over_token_limit'] += 1
                    continue
                pairs[split].append(dict(id=row['id'],family_id=row['family_id'],question=row['question'],
                    conversations=row['conversations'][:-1],chosen=chosen,rejected=rejected,
                    reference=row['answer'],rejected_value=tool['value']))
            print(dict(counts)|{'train_pairs':len(pairs['train']),'valid_pairs':len(pairs['valid'])},flush=True)
            if len(pairs['train']) == args.train_size and len(pairs['valid']) == args.valid_size:
                break
        if len(pairs['train']) < args.train_size or len(pairs['valid']) < args.valid_size:
            raise RuntimeError('Insufficient executable wrong responses; sampling records preserved')
        if {r['family_id'] for r in pairs['train']} & {r['family_id'] for r in pairs['valid']}:
            raise ValueError('Preference family leakage')
        for split,data in pairs.items():
            write_lines(directory/f'{split}/pairs.jsonl',data)
        atomic_json(directory/'quality_report.json',dict(counts)|{'train_pairs':len(pairs['train']),
            'valid_pairs':len(pairs['valid']),'limitation':'Numeric wrongness relative to reference; explanation semantics not individually human verified.'})


if __name__ == '__main__':
    main()
