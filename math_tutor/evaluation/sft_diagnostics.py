"""Offline SFT arithmetic diagnostics. Never changes generated answers or scores."""
import argparse
import hashlib
import math
import re
from collections import Counter
from math_tutor.common import ROOT, read_records, dump, write_lines
from math_tutor.data.prepare import expression
from math_tutor.evaluation.scoring import numeric_fragment


def inspect_response(text):
    equations = re.findall(r'^\s*列式\s*[:：]\s*(.+)$', text, re.M)
    answers = re.findall(r'^\s*答案\s*[:：]\s*(.+)$', text, re.M)
    if len(equations)!=1 or len(answers)!=1:
        return dict(arithmetic_status='needs_review', reason='Expected exactly one equation and answer')
    answer=numeric_fragment(answers[0])
    if answer is None:
        return dict(arithmetic_status='uncheckable', reason='Answer cannot be parsed')
    try:
        _, value=expression(equations[0])
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
        return dict(arithmetic_status='uncheckable', reason='Unsupported or invalid expression')
    ok=math.isclose(value,answer,rel_tol=1e-5,abs_tol=1e-5)
    return dict(arithmetic_status='consistent' if ok else 'inconsistent',
                equation_value=value, answer_value=answer,
                reason='Strict numeric tolerance; rounded answers may need human review. Does not validate the equation against the question.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/sft-diagnostics-v1')
    p.add_argument('--review',default='data/reviews/review30.readable.json')
    args=p.parse_args(); out=ROOT/args.output
    if out.exists(): raise SystemExit('Choose a new report directory')
    evaluations=read_records(ROOT/'results/sft-valid.jsonl')
    references={r['id']:r for r in read_records(ROOT/'data/processed/valid.jsonl')}
    by_id={r['id']:r for r in evaluations}
    reviews=read_records(ROOT/args.review)
    if len({r['id'] for r in reviews})!=len(reviews): raise ValueError('Duplicate review IDs')
    warnings=[]
    for r in reviews:
        original=by_id[r['id']]
        if r['question']!=original['question'] or r['model_response']!=original['response'] or r['reference_answer']!=original['reference']:
            raise ValueError('Review content differs from evaluation: '+r['id'])
        if r.get('reviewed') is not True: warnings.append(r['id']+': not reviewed')
        if r.get('primary_error') not in {'none','reference','formulation','arithmetic','contradiction','format','scoring','unsure'}:
            warnings.append(r['id']+': unknown error category')
    checked=[dict(r, **inspect_response(r['response'])) for r in evaluations]
    write_lines(out/'automatic_checks.jsonl',checked)
    remaining=[]
    reviewed_ids={r['id'] for r in reviews if r.get('reviewed') is True}
    for r in evaluations:
        if r['id'] in reviewed_ids: continue
        remaining.append(dict(id=r['id'],split='valid',question=r['question'],
            reference_equation=references[r['id']]['equation'],reference_answer=r['reference'],
            model_response=r['response'],auto_status=r['status'],auto_check=inspect_response(r['response']),
            reviewed=False,reference_ok=None,equation_matches_question=None,calculation_consistent=None,
            final_answer_correct=None,format_ok=None,primary_error='',notes='',reviewer=''))
    dump(out/'remaining_review.readable.json',remaining)
    # Report disagreement for manual attention; no automatic relabelling.
    disagreement=[]
    for r in reviews:
        check=inspect_response(r['model_response'])
        expected={'consistent':True,'inconsistent':False}.get(check['arithmetic_status'])
        if expected is not None and r.get('calculation_consistent') is not None and r['calculation_consistent']!=expected:
            disagreement.append(dict(id=r['id'],human=r['calculation_consistent'],automatic=check))
    dump(out/'review_disagreements.json',disagreement)
    summary=dict(n=len(checked), arithmetic_counts=dict(Counter(r['arithmetic_status'] for r in checked)),
        reviewed=len(reviewed_ids),remaining=len(remaining),
        human_error_counts=dict(Counter(r.get('primary_error') for r in reviews)),warnings=warnings,
        disagreement_count=len(disagreement),
        note='Human queue is deliberately sampled, not a population error-rate estimate. Arithmetic checks do not change original accuracy. Rounding can cause disagreement.')
    dump(out/'summary.json',summary)
    dump(out/'input_hashes.json', {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['results/sft-valid.jsonl','data/processed/valid.jsonl',args.review]})
    print(summary)

if __name__=='__main__': main()
