"""Offline arithmetic diagnostics. Never changes generated answers or scores."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import re
import unicodedata
from math_tutor.common import ROOT, read_records, dump, write_lines
from math_tutor.data.prepare import expression
from math_tutor.evaluation.scoring import numeric_fragment


def diagnose(response, reference):
    # Preserve ** exponentiation; scorer.normalize is intended for answer text only.
    text = unicodedata.normalize('NFKC', response).replace('−', '-')
    formulas = re.findall(r'^\s*列式\s*:\s*(.*?)\s*$', text, re.M)
    answers = re.findall(r'^\s*答案\s*:\s*(.*?)\s*$', text, re.M)
    result = dict(formula_count=len(formulas), answer_count=len(answers),
                  status='uncheckable', arithmetic_consistent=None,
                  expression_matches_reference_value=None)
    if len(formulas)>1 or len(answers)>1:
        return dict(result, status='multiple_outputs')
    if len(formulas)!=1 or len(answers)!=1:
        return result
    try:
        normalized, value = expression(formulas[0])
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return result
    answer = numeric_fragment(answers[0])
    if answer is None:
        return result
    consistent = math.isclose(value, answer, rel_tol=1e-5, abs_tol=1e-5)
    return dict(result, status='consistent' if consistent else 'inconsistent',
        arithmetic_consistent=consistent, expression=normalized,
        expression_value=value, emitted_answer=answer,
        expression_matches_reference_value=math.isclose(value,float(reference),rel_tol=1e-5,abs_tol=1e-5),
        note='Strict numeric tolerance. Rounded values may require review. Matching reference value does not prove correct formulation.')


def review_record(row, reference, auto):
    return dict(id=row['id'], split='valid', question=row['question'],
        reference_equation=reference['equation'], reference_answer=row['reference'],
        model_response=row['response'], auto_status=row['status'], arithmetic_diagnostic=auto,
        reviewed=False, reference_ok=None, equation_matches_question=None,
        calculation_consistent=None, final_answer_correct=None, format_ok=None,
        primary_error='', notes='', reviewer='')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='reports/sft-diagnosis-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--reviews', default='data/reviews/review30.corrected.json')
    args=p.parse_args(); out=ROOT/args.output
    if out.exists(): raise SystemExit('Choose a new output directory')
    source=ROOT/'results/sft-valid.jsonl'
    rows=read_records(source)
    refs={r['id']:r for r in read_records(ROOT/'data/processed/valid.jsonl')}
    if len({r['id'] for r in rows})!=len(rows) or set(refs)!={r['id'] for r in rows}:
        raise ValueError('Duplicate or mismatched validation IDs')
    review_path=ROOT/args.reviews
    reviews=json.loads(review_path.read_text(encoding='utf-8-sig'))
    lookup={r['id']:r for r in rows}
    if len({r['id'] for r in reviews})!=len(reviews): raise ValueError('Duplicate review IDs')
    for r in reviews:
        if r['id'] not in lookup or r['model_response']!=lookup[r['id']]['response'] or r['question']!=lookup[r['id']]['question']:
            raise ValueError('Review source mismatch')
    reviewed={r['id'] for r in reviews if r.get('reviewed') is True}
    outputs=[]; remaining=[]
    for r in rows:
        if r['question']!=refs[r['id']]['question'] or r['reference']!=refs[r['id']]['answer']:
            raise ValueError('Validation content mismatch')
        auto=diagnose(r['response'],r['reference'])
        outputs.append(dict(r,arithmetic_diagnostic=auto))
        if r['id'] not in reviewed: remaining.append(review_record(r,refs[r['id']],auto))
    counts=Counter(r['arithmetic_diagnostic']['status'] for r in outputs)
    checked=counts['consistent']+counts['inconsistent']
    summary=dict(n=len(rows),status_counts=dict(counts),checked=checked,
        coverage=checked/len(rows) if rows else None,
        consistency_among_checked=counts['consistent']/checked if checked else None,
        human_reviewed=len(reviewed),remaining=len(remaining),
        human_sample_types=dict(Counter(r['primary_error'] for r in reviews if r.get('reviewed') is True)),
        limitations=['Arithmetic only; formulation needs human verification.',
                    '30-case human sample is intentionally biased; do not extrapolate its proportions.',
                    'Original model outputs and original accuracy are unchanged.'])
    dump(out/'summary.json',summary)
    write_lines(out/'diagnostics.jsonl',outputs)
    dump(out/'review_remaining.readable.json',remaining)
    dump(out/'inputs.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source, review_path, ROOT/'data/processed/valid.jsonl')})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
