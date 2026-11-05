"""Offline data audit, paired comparisons and human review queue. No GPU needed."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from math_tutor.common import ROOT, read_records, dump, write_lines


def indexed(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate evaluation IDs')
    return result


def compare(left, right):
    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError('Evaluation ID sets differ; partial comparisons are forbidden')
    counts = Counter()
    changed = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if (x['question'], x['reference']) != (y['question'], y['reference']):
            raise ValueError('Question/reference mismatch: ' + key)
        # Unknown statuses are not labelled as confirmed mathematical errors.
        xc, yc = x.get('correct') is True, y.get('correct') is True
        counts['both_confirmed_correct' if xc and yc else
               'gained_confirmed_correct' if yc else
               'lost_confirmed_correct' if xc else 'neither_confirmed_correct'] += 1
        if x['response'] != y['response']:
            counts['response_changed'] += 1
        if x['response'] != y['response'] or x.get('status') != y.get('status'):
            changed.append(dict(id=key, question=x['question'], reference=x['reference'],
                left_response=x['response'], right_response=y['response'],
                left_status=x.get('status'), right_status=y.get('status')))
    counts['n'] = len(a)
    counts['accuracy_delta_pp'] = 100 * (counts['gained_confirmed_correct'] - counts['lost_confirmed_correct']) / len(a) if a else None
    return dict(counts), changed


def data_audit(root):
    splits = {s: read_records(root / f'data/processed/{s}.jsonl') for s in ('train', 'valid', 'test')}
    info = {}
    for split, rows in splits.items():
        info[split] = dict(count=len(rows), duplicate_ids=len(rows)-len({r['id'] for r in rows}),
            duplicate_families=len(rows)-len({r['family_id'] for r in rows}))
    overlap = {}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlap[a+'__'+b] = len({r['family_id'] for r in splits[a]} & {r['family_id'] for r in splits[b]})
    return dict(splits=info, family_overlap=overlap,
        passed=not any(overlap.values()) and all(v['duplicate_ids']==0 and v['duplicate_families']==0 for v in info.values()),
        limitation='Family checks are heuristic; semantic duplicates and question correctness need human review.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/audit-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise SystemExit('Choose a new output directory; reports are never overwritten')
    comparisons = [('base-test', 'sft-test'), ('sft-valid', 'merged-sft-valid'),
                   ('merged-sft-valid', 'dpo-demo-valid')]
    evidence, queue, lines = {}, [], ['# 自动项目审计', '', '仅使用现有文件；没有重新推理或训练。', '']
    audit = data_audit(ROOT)
    dump(out / 'data_audit.json', audit)
    lines += ['## 数据检查', '', '```json', json.dumps(audit, ensure_ascii=False, indent=2), '```', '']
    manifest = {}
    for path in sorted((ROOT/'results').glob('*.jsonl')):
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for left, right in comparisons:
        paths = [ROOT/f'results/{name}.jsonl' for name in (left, right)]
        if not all(p.exists() for p in paths):
            evidence[left+'__'+right] = {'skipped': 'Missing evaluation file'}
            continue
        summaries = [json.loads(p.with_suffix('.summary.json').read_text(encoding='utf-8-sig')) for p in paths]
        condition_keys = ['data_sha256', 'system_prompt', 'max_new_tokens', 'do_sample', 'scorer_version', 'n']
        mismatch = [k for k in condition_keys if any(k not in s for s in summaries) or summaries[0].get(k) != summaries[1].get(k)]
        if mismatch:
            evidence[left+'__'+right] = {'blocked': mismatch}
            continue
        stats, changed = compare(*(read_records(p) for p in paths))
        evidence[left+'__'+right] = stats
        write_lines(out/f'{left}__{right}.changed.jsonl', changed)
        lines += [f'## {left} → {right}', '', '```json', json.dumps(stats, indent=2), '```', '']
    # Validation only: keep test-set errors away from preference-data construction.
    valid = indexed(read_records(ROOT/'results/dpo-demo-valid.jsonl')) if (ROOT/'results/dpo-demo-valid.jsonl').exists() else {}
    for row in sorted(valid.values(), key=lambda r: (r.get('correct') is True, r['id'])):
        queue.append(dict(id=row['id'], split='valid', question=row['question'], reference=row['reference'],
            response=row['response'], auto_status=row.get('status'), reviewed=False,
            reference_correct=None, solution_correct=None, final_answer_correct=None, unit_correct=None,
            error_type=None, corrected_response=None, reviewer=None, notes='',
            usage='Evaluation review only; do not put validation cases into training data.'))
    write_lines(out/'human_review.valid.jsonl', queue)
    dump(out/'comparisons.json', evidence)
    dump(out/'input_hashes.json', manifest)
    lines += ['## 解释边界', '', '自动数值正确不代表推导正确。数据分组不证明无语义泄漏。',
        '审阅队列只用于验证集诊断，不可直接转成训练偏好对。',
        '比较 SFT 合并前后的差异时，需用同一提示、精度和生成参数重新确认。', '']
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
