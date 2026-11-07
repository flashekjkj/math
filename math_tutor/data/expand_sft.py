"""Build a nested 3000-row experiment, preserving old examples and prompts."""
import argparse
import hashlib
import json
import random
from math_tutor.common import ROOT, read_records, write_lines, dump
from math_tutor.data.prepare import convert

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/versions/sft-scale-3k-v1')
    p.add_argument('--config',default='configs/sft-scale-3k-v1.json')
    args=p.parse_args(); out=ROOT/args.output; config_path=ROOT/args.config
    if out.exists() or config_path.exists(): raise SystemExit('Version exists; choose new paths')
    old={s:read_records(ROOT/f'data/processed/{s}.jsonl') for s in ('train','valid','test')}
    if len(old['train'])!=1000: raise ValueError('Expected frozen 1000-row baseline')
    systems={m['value'] for r in old['train'] for m in r['conversations'] if m['from']=='system'}
    if len(systems)!=1: raise ValueError('Mixed baseline prompts; resolve before expansion')
    system=next(iter(systems))
    raw=ROOT/'data/raw/math23k_train.json'; test_raw=ROOT/'data/raw/math23k_test.json'
    clean,rejects=convert(read_records(raw),'train')
    official_test,_=convert(read_records(test_raw),'test')
    # Exclude all official test families, not only the selected 200 test rows.
    seen={r['family_id'] for r in official_test}
    seen.update(r['family_id'] for split in old.values() for r in split)
    old_ids={r['id'] for split in old.values() for r in split}
    candidates=[]
    for r in clean:
        if r['id'] in old_ids or r['family_id'] in seen: continue
        # Preserve the original hash-defined validation pool, including unused rows.
        if int(r['family_id'][:8],16)%10==0: continue
        seen.add(r['family_id']); candidates.append(r)
    random.Random(42).shuffle(candidates)
    added=candidates[:2000]
    if len(added)!=2000: raise ValueError('Not enough eligible training families')
    for r in added:
        for m in r['conversations']:
            if m['from']=='system': m['value']=system
    train=old['train']+added
    if len({r['family_id'] for r in train})!=3000: raise ValueError('Duplicate training families')
    for split,rows in dict(train=train,valid=old['valid'],test=old['test']).items():
        write_lines(out/f'processed/{split}.jsonl',rows)
        if split!='test': write_lines(out/f'sft_{split}/{split}.jsonl',[{'conversations':r['conversations']} for r in rows])
    selected=random.Random(43).sample(added,100)
    dump(out/'added100.review.readable.json',[dict(id=r['id'],question=r['question'],equation=r['equation'],
        answer=r['answer'],target=r['conversations'][-1]['value'],reviewed=False,decision='',reason='',reviewer='') for r in selected])
    dump(out/'manifest.json',dict(original_train=1000,added=2000,total=3000,seed=42,
        added_ids=[r['id'] for r in added],raw_sha256=sha(raw),official_test_sha256=sha(test_raw),
        frozen_inputs={s:sha(ROOT/f'data/processed/{s}.jsonl') for s in old},
        system_prompt=system, rejected_source_count=len(rejects),
        warning='Candidate experiment only; 2000 added records not semantically reviewed. Audit token lengths and review sample before training. Same epoch means more updates than baseline.'))
    c=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8-sig'))
    c.update(train_file_dir=f'{args.output}/sft_train',validation_file_dir=f'{args.output}/sft_valid',output_dir='runs/sft-scale-3k-v1')
    dump(config_path,c)
    print(out); print(config_path)

if __name__=='__main__': main()
