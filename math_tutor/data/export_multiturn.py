"""Export only fully audited samples with a fixed 50/30/20 conversation ratio."""
import hashlib,json,random
from collections import Counter
from transformers import AutoTokenizer
from math_tutor.common import ROOT,read_records,write_lines,dump
from math_tutor.data.multiturn_data import SYSTEM
from math_tutor.data.prepare import expression
from math_tutor.training.multiturn_sft import encode_conversation

def main():
    out=ROOT/'data/versions/multiturn-v1'
    source=read_records(ROOT/'data/versions/teacher-analysis-v1/accepted.jsonl')
    families=sorted({r['family_id'] for r in source});random.Random(9127).shuffle(families);held=set(families[:80])
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'records').glob('*.json'))]
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen-0.5b');allrows={};stats={};lengths=[]
    for split,maxunits,seed in [('train',60,42),('valid',8,43)]:
        pool=[r for r in source if (r['family_id'] in held)==(split=='valid')];random.Random(seed).shuffle(pool)
        generated=[r for r in records if r['status']=='accepted' and r['split']==split]
        mathrows=[r for r in generated if r['kind']!='chat' and r.get('context_audit',{}).get('valid') is True]
        chat=[r for r in generated if r['kind']=='chat']
        units=min(maxunits,len(mathrows)//3,len(chat)//2)
        if units<6:raise RuntimeError('Insufficient audited data')
        nsingle,nmulti,nchat=5*units,3*units,2*units
        single=[]
        for row in pool[:nsingle]:
            _,value=expression(row['expression'])
            single.append({'id':row['id'],'family_id':row['family_id'],'kind':'single','conversations':[{'from':'system','value':SYSTEM},{'from':'human','value':row['question']},{'from':'gpt','value':f'分析：{row["analysis"]}\n列式：{row["expression"]}\n答案：{value:.10g}'}]})
        rows=single+mathrows[:nmulti]+chat[:nchat];random.Random(seed).shuffle(rows)
        for row in rows:
            encoded=encode_conversation(tok,row['conversations']);lengths.append(len(encoded['input_ids']))
        allrows[split]=rows
        stats[split]={'total':len(rows),'single':nsingle,'multi':nmulti,'chat':nchat,'assistant_turns':nsingle+3*(nmulti+nchat),'kinds':dict(Counter(r['kind'] for r in rows))}
        write_lines(out/f'sft_{split}/{split}.jsonl',[{'conversations':r['conversations']} for r in rows]);write_lines(out/f'{split}.metadata.jsonl',rows)
    assert not {r['family_id'] for r in allrows['train']} & {r['family_id'] for r in allrows['valid']}
    (out/'system_prompt.txt').write_text(SYSTEM,encoding='utf-8')
    dump(out/'manifest.json',{'ready_for_training':True,'counts':stats,'max_tokens':max(lengths),'min_tokens':min(lengths),'family_disjoint':True,'conversation_ratio':[0.5,0.3,0.2],'native_token_mask_verified':True,'source_sha256':hashlib.sha256((ROOT/'data/versions/teacher-analysis-v1/accepted.jsonl').read_bytes()).hexdigest(),'exports_sha256':{s:hashlib.sha256((out/f'sft_{s}/{s}.jsonl').read_bytes()).hexdigest() for s in allrows},'limitations':'Synthetic labels, same teacher separate reviews, no human gold. New SFT family split is disjoint; old SFT had seen these source families. Chat semantic topics may overlap. Ratios count conversations, not tokens. Export count capped by accepted audited candidates.'})
    print(stats,'max tokens',max(lengths),flush=True)
if __name__=='__main__':main()
