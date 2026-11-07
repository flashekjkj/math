"""Compare new system results with preserved historical SFT results by ID."""
import hashlib
from collections import Counter
from math_tutor.common import ROOT,read_records,dump

def main():
    paths={'old_sft_raw':'results/sft-test.jsonl','old_sft_calculator':'results/old-sft-calculator-test.jsonl',
           'mechanical_relation_tool':'results/relation-tool-test.jsonl','teacher_analysis_calculator':'results/analysis-sft-test.jsonl'}
    groups={name:read_records(ROOT/path) for name,path in paths.items()}
    indexed={name:{r['id']:r for r in rows} for name,rows in groups.items()}
    reference=indexed['old_sft_raw'];n=len(reference)
    for name,rows in indexed.items():
        if len(rows)!=len(groups[name]) or rows.keys()!=reference.keys():raise ValueError('Mismatched/duplicate IDs')
        for key,row in rows.items():
            if 'question' in row and (row['question'],row['reference'])!=(reference[key]['question'],reference[key]['reference']):raise ValueError('Question/reference mismatch')
    stats={name:dict(n=n,correct=sum(r.get('correct') is True for r in rows.values()),accuracy=sum(r.get('correct') is True for r in rows.values())/n) for name,rows in indexed.items()}
    new=indexed['teacher_analysis_calculator'];old=indexed['old_sft_calculator']
    paired=Counter()
    for key in reference:
        a,b=old[key]['correct'] is True,new[key]['correct'] is True
        paired['both_correct' if a and b else 'gain' if b else 'regression' if a else 'neither_correct']+=1
    dump(ROOT/'results/analysis-sft-comparison.json',dict(results=stats,paired_vs_old_calculator=dict(paired),
        delta_vs_old_calculator_pp=100*(stats['teacher_analysis_calculator']['accuracy']-stats['old_sft_calculator']['accuracy']),
        inputs={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths.values()},
        limitations=['Historical baseline generations reused; generation batch settings differ.',
                    '954 filtered teacher-labelled records vs old 1000 records; not a pure analysis-only ablation.',
                    'System accuracy includes calculator; not independent arithmetic or explanation correctness.',
                    'Previously observed test set, not new blind testing.']))
    print(stats);print(dict(paired))

if __name__=='__main__':main()
