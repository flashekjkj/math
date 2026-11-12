"""Conservative numeric and expression checks, separate from semantic review."""
import argparse
from collections import Counter
import math
import re
import unicodedata

from math_tutor.common import ROOT, read_records, write_lines
from math_tutor.data.prepare import expression
from math_tutor.evaluation.scoring import score_response
from math_tutor.data.preference_io import atomic_json, sha256


def check_candidate(question, reference, candidate):
    score = score_response(question, candidate['text'], reference)
    text = unicodedata.normalize('NFKC', candidate['text'])
    matches = re.findall(r'^\s*列式\s*[:：]\s*(.+)$', text, re.MULTILINE)
    checks = []
    for raw in matches:
        try:
            normalized, value = expression(raw.strip())
            checks.append(dict(raw=raw, parseable=True, normalized=normalized, value=value,
                matches_reference=math.isclose(value,float(reference),rel_tol=1e-5,abs_tol=1e-5),
                matches_output=None if score['prediction'] is None else
                    math.isclose(value,score['prediction'],rel_tol=1e-5,abs_tol=1e-5)))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
            checks.append(dict(raw=raw, parseable=False))
    flags = []
    if candidate['finish_reason'] == 'length':
        flags.append('truncated')
    if score['status'] in ('unparsed','conflict','needs_review'):
        flags.append(score['status'])
    if not matches:
        flags.append('no_explicit_expression')
    if any(not e['parseable'] for e in checks):
        flags.append('unsupported_expression')
    if any(e.get('matches_output') is False for e in checks):
        flags.append('expression_answer_mismatch')
    return dict(numeric=score, expressions=checks, flags=flags, semantic_status='pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/versions/preference-sample-v1')
    args = parser.parse_args()
    directory = ROOT / args.input
    source = directory / 'samples.jsonl'
    records = read_records(source)
    statuses = Counter()
    flags = Counter()
    output = []
    for row in records:
        candidates = []
        for candidate in row['candidates']:
            checks = check_candidate(row['question'], row['answer'], candidate)
            statuses[checks['numeric']['status']] += 1
            flags.update(checks['flags'])
            candidates.append(candidate | {'automatic':checks})
        output.append(row | {'candidates':candidates})
    write_lines(directory / 'checked.jsonl', output)
    summary = dict(source_sha256=sha256(source), questions=len(records), status_counts=dict(statuses),
                   flags=dict(flags), mixed_numeric_groups=sum(
                   any(c['automatic']['numeric']['correct'] is True for c in r['candidates']) and
                   any(c['automatic']['numeric']['correct'] is False for c in r['candidates']) for r in output))
    atomic_json(directory / 'automatic_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
