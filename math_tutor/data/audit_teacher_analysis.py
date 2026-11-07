"""Finalize matched control data and audit teacher outputs without API calls."""
import hashlib
import json
from collections import Counter
from math_tutor.common import ROOT,read_records,write_lines,dump
from math_tutor.data.teacher_analysis import validate, STUDENT_SYSTEM

def main():
    out=ROOT/'data/versions/teacher-analysis-v1'
    accepted=read_records(out/'accepted.jsonl')
    sources={r['id']:r for r in read_records(ROOT/'data/processed/train.jsonl')}
    if len({r['id'] for r in accepted})!=len(accepted):raise ValueError('Duplicate accepted IDs')
    for r in accepted:
        record=json.loads((out/'records'/(r['id'].replace(':','_')+'.json')).read_text(encoding='utf-8'))
        if record['review'].get('valid') is not True:raise ValueError('Missing semantic review')
        analysis,eq=validate(record['teacher'],sources[r['id']])
        if (analysis,eq)!=(r['analysis'],r['expression']):raise ValueError('Export mismatch')
    write_lines(out/'control_train/train.jsonl',[{'conversations':[
        {'from':'system','value':'你是数学助手。根据题意建立数量关系，仅输出一行“列式：数学表达式”。不要自己计算最终答案，结果由外部计算器执行。'},
        {'from':'human','value':r['question']},{'from':'gpt','value':'列式：'+r['expression']}]} for r in accepted])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen-0.5b'),trust_remote_code=False)
    lengths=[]
    role={'system':'system','human':'user','gpt':'assistant'}
    for r in accepted:
        messages=[{'role':role[m['from']],'content':m['value']} for m in r['conversations']]
        lengths.append(len(tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False)))
    usage=json.loads((out/'usage.json').read_text(encoding='utf-8'))
    # Only run offline after the generator has exited. Unfinished historic calls
    # retain their reservation; their actual billing cannot be inferred locally.
    for request in usage['requests']:
        if request['status']=='pending':request['status']='interrupted_unknown'
    dump(out/'usage.json',usage)
    report=dict(accepted=len(accepted),analysis_char_min=min((len(r['analysis']) for r in accepted),default=0),
        analysis_char_max=max((len(r['analysis']) for r in accepted),default=0),token_max=max(lengths,default=0),
        token_p95=sorted(lengths)[min(len(lengths)-1,int(len(lengths)*.95))] if lengths else None,
        over_384=sum(n>384 for n in lengths),provider_reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in usage['requests']),
        charged_or_reserved_tokens=usage['charged_or_reserved_tokens'],requests=len(usage['requests']),
        request_status_counts=dict(Counter(r['status'] for r in usage['requests'])),
        export_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (out/'accepted.jsonl',out/'sft_train/train.jsonl',out/'control_train/train.jsonl')})
    dump(out/'audit.json',report)
    (out/'system_prompt.txt').write_text(STUDENT_SYSTEM+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
