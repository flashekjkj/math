"""Construct synthetic numerical preference pairs for pipeline verification only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from math_tutor.common import ROOT, SYSTEM, read_records, write_lines, dump
from math_tutor.data.prepare import expression

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/versions/preference-demo-v1')
    p.add_argument('--train-size',type=int,default=200)
    p.add_argument('--valid-size',type=int,default=50)
    args=p.parse_args()
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()): raise SystemExit('Output exists: choose a new version')
    frozen={r['family_id'] for r in read_records(ROOT/'data/processed/test.jsonl')}
    groups={};data={};sources={}
    for split,limit in [('train',args.train_size),('valid',args.valid_size)]:
        source=ROOT/f'data/processed/{split}.jsonl'
        rows=[]
        for row in read_records(source):
            fid=row['family_id']
            if fid in frozen: raise ValueError('Test-family leakage')
            if fid in groups and groups[fid]!=split: raise ValueError('Train/valid family overlap')
            groups[fid]=split
            equation,value=expression(row['equation'])
            if not math.isclose(value,row['answer'],rel_tol=1e-6,abs_tol=1e-6): continue
            wrong=value+max(1,abs(value)*.1)
            rows.append({'conversations':[{'from':'system','value':SYSTEM},{'from':'human','value':row['question']}],
                'chosen':f'列式：{equation}\n答案：{value:.10g}',
                'rejected':f'列式：{equation}\n答案：{wrong:.10g}'})
            if len(rows)>=limit: break
        if not rows: raise ValueError('No valid pairs')
        data[split]=rows
        sources[split]={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'count':len(rows)}
    for split,rows in data.items(): write_lines(out/split/'pairs.jsonl',rows)
    dump(out/'manifest.json',{'sources':sources,'kind':'synthetic_numerical_consistency_demo',
        'limitations':'Only equation/answer consistency checked. These are not human teaching preferences or independently verified question solutions.'})
    print(json.dumps({k:len(v) for k,v in data.items()}))
if __name__=='__main__': main()
