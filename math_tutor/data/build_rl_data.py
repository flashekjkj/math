"""Export exactly the questions used by the existing DPO preference run."""
import hashlib
from math_tutor.common import ROOT, read_records, write_lines, dump

def main():
    out = ROOT / 'data/versions/rl-v1'
    if out.exists():
        raise SystemExit('RL data already exists; choose a new version before rebuilding')
    frozen = {r['family_id'] for r in read_records(ROOT/'data/processed/test.jsonl')}
    groups = {}
    sources = {}
    exported = {}
    for split in ('train', 'valid'):
        src = ROOT/f'data/versions/preference-demo-v1/{split}/pairs.jsonl'
        lookup = {r['question']: r for r in read_records(ROOT/f'data/processed/{split}.jsonl')}
        rows = []
        for pair in read_records(src):
            messages = [{'role': {'system':'system','human':'user'}[m['from']], 'content':m['value']}
                        for m in pair['conversations']]
            q = messages[-1]['content']
            record = lookup[q]
            fid = record['family_id']
            if fid in frozen or (fid in groups and groups[fid] != split):
                raise ValueError('Split leakage')
            groups[fid] = split
            rows.append({k:record[k] for k in ('id','family_id','question','answer','equation')} | {'prompt':messages})
        exported[split] = rows
        sources[split] = {'preference_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'count':len(rows)}
    for split, rows in exported.items():
        write_lines(out/f'{split}/{split}.jsonl', rows)
    dump(out/'manifest.json', {'sources':sources,'note':'Prompts exactly match DPO pairs; answer/equation are reward metadata, not model input.'})

if __name__ == '__main__':
    main()
