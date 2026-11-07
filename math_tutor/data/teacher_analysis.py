"""Resumable teacher generation + independent semantic review, training split only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import threading
import urllib.request
import urllib.error
from math_tutor.common import ROOT, read_records, write_lines, dump
from math_tutor.data.prepare import expression

TEACHER = '''你是小学应用题标注老师。独立核查题意、参考列式和答案，再生成简短教学分析。只返回JSON对象：{"valid":true,"analysis":"...","expression":"...","reason":""}。
analysis为40至140个中文字符左右的1至3句：明确所求量，说明关键数量关系、比较基准和运算顺序；必须结合本题含义，不要仅复述算式，不要无关套话，不要计算最终数值。expression仅含可执行四则表达式、数字、括号，百分数写成除以100。可以使用等价于参考的正确式子。
若缺图、条件不足、题意歧义、多个问题仅有一个参考答案、参考不合理，valid=false并在reason说明，不要强行解释参考答案。'''
REVIEWER = '''你是独立数学质检员。检查题目、分析和列式，不要仅因计算结果相同就认定正确。检查题意是否完整且单问、分析是否解释实际数量关系、有没有编造条件、比较基准是否正确、分析与列式是否一致。只返回JSON：{"valid":true或false,"reason":"简短理由"}。不确定就false。'''
STUDENT_SYSTEM='你是中文数学助手。先用1至3句简短分析说明题意与数量关系，再给出列式。仅输出“分析：...”和“列式：...”两部分。不计算最终答案，算式由外部计算器执行。'

def validate(obj,row):
    if obj.get('valid') is not True:raise ValueError('teacher_rejected: '+str(obj.get('reason',''))[:160])
    analysis=obj.get('analysis');eq=obj.get('expression')
    if not isinstance(analysis,str) or not 25<=len(analysis)<=160 or '\n' in analysis:
        raise ValueError('analysis_length_or_format')
    if not isinstance(eq,str):raise ValueError('expression_type')
    normalized,value=expression(eq)
    if not math.isclose(value,row['answer'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('answer_mismatch')
    if any(s in analysis for s in ('所求量按','括号内先计算','答案：','最终答案')):raise ValueError('mechanical_or_answer_leak')
    return analysis,normalized

class Client:
    def __init__(self,out,url,key,model,budget):
        self.out,self.url,self.key,self.model,self.budget=out,url,key,model,budget
        self.lock=threading.Lock();self.halt=False
        self.state_path=out/'usage.json'
        self.state=json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {'charged_or_reserved_tokens':0,'requests':[]}
        for request in self.state['requests']:
            if request['status']=='pending':request['status']='interrupted_unknown'
    def save(self):
        temp=self.state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,self.state_path)
    def call(self,system,payload,record_id,stage):
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        reserve=len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))+512+128
        with self.lock:
            if self.halt or self.state['charged_or_reserved_tokens']+reserve>self.budget:raise RuntimeError('budget_or_remote_stop')
            index=len(self.state['requests']);self.state['charged_or_reserved_tokens']+=reserve
            self.state['requests'].append(dict(id=record_id,stage=stage,reservation=reserve,status='pending'))
            self.save()
        body=json.dumps(dict(model=self.model,messages=messages,max_tokens=512,temperature=0.2,enable_thinking=False,stream=False),ensure_ascii=False).encode('utf-8')
        request=urllib.request.Request(self.url.rstrip('/')+'/chat/completions',data=body,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=120) as response: result=json.load(response)
        except urllib.error.HTTPError as exc:
            with self.lock:
                self.state['requests'][index]['status']='http_'+str(exc.code);self.halt=True;self.save()
            raise RuntimeError('HTTP '+str(exc.code)+'; stopped without printing remote body or credentials') from None
        except Exception:
            with self.lock:
                self.state['requests'][index]['status']='transport_unknown';self.halt=True;self.save()
            raise RuntimeError('Transport failure; reservation retained; no automatic retry') from None
        usage=result.get('usage',{});total=usage.get('total_tokens')
        with self.lock:
            entry=self.state['requests'][index];entry.update(status='received',usage=usage,response_id=result.get('id'),returned_model=result.get('model'))
            if isinstance(total,int) and total>=0:self.state['charged_or_reserved_tokens']+=total-reserve
            else:self.halt=True
            if self.state['charged_or_reserved_tokens']>self.budget:self.halt=True
            self.save()
        choice=result['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('truncated_or_nonstop_response')
        text=choice['message'].get('content','').strip()
        if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
        return json.loads(text)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default=os.getenv('DASHSCOPE_BASE_URL','https://maas.qianwenaiapi.com/compatible-mode/v1'))
    p.add_argument('--model',default='deepseek-v4.1-flash')
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--budget',type=int,default=800000)
    p.add_argument('--output',default='data/versions/teacher-analysis-v1')
    args=p.parse_args()
    if not 1<=args.workers<=4 or not 1<=args.budget<=850000:raise SystemExit('workers 1..4; budget <=850000')
    if not args.base_url.startswith('https://'):raise SystemExit('HTTPS required')
    key=os.getenv('DASHSCOPE_API_KEY')
    if not key:raise SystemExit('Set DASHSCOPE_API_KEY in process environment; never put it in source')
    source=ROOT/'data/processed/train.jsonl'; rows=read_records(source)
    frozen=set()
    for split in ('valid','test'):frozen.update(r['family_id'] for r in read_records(ROOT/f'data/processed/{split}.jsonl'))
    if any(r['family_id'] in frozen for r in rows):raise ValueError('Split leakage')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    meta=dict(model=args.model,base_url=args.base_url,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prompt_sha256=hashlib.sha256((TEACHER+REVIEWER).encode()).hexdigest())
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text(encoding='utf-8'))!=meta:raise SystemExit('Source/prompt/model changed; use new version')
    dump(out/'provenance.json',meta)
    client=Client(out,args.base_url,key,args.model,args.budget)
    records=out/'records';records.mkdir(exist_ok=True)
    def work(row):
        path=records/(row['id'].replace(':','_')+'.json')
        if path.exists():return
        item=dict(id=row['id'],question=row['question'],reference_equation=row['equation'],reference_answer=row['answer'])
        try:
            draft=client.call(TEACHER,dict(question=row['question'],reference_equation=row['equation'],reference_answer=row['answer']),row['id'],'generate')
            item['teacher']=draft
            analysis,eq=validate(draft,row)
            verdict=client.call(REVIEWER,dict(question=row['question'],analysis=analysis,expression=eq),row['id'],'review')
            item['review']=verdict
            if verdict.get('valid') is not True:raise ValueError('review_rejected')
            item.update(status='accepted',analysis=analysis,expression=eq)
        except RuntimeError as exc:
            print(row['id'],str(exc),flush=True);return
        except Exception as exc:
            item.update(status='rejected',error=str(exc)[:200])
        dump(path,item)
        print(row['id'],item['status'],flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(work,rows[:args.limit]))
    accepted=[]; rejected=[]
    for row in rows:
        path=records/(row['id'].replace(':','_')+'.json')
        if not path.exists():continue
        r=json.loads(path.read_text(encoding='utf-8'))
        if r['status']=='accepted':accepted.append(dict(id=r['id'],family_id=row['family_id'],analysis=r['analysis'],expression=r['expression'],question=row['question'],conversations=[{'from':'system','value':STUDENT_SYSTEM},{'from':'human','value':row['question']},{'from':'gpt','value':f'分析：{r["analysis"]}\n列式：{r["expression"]}'}]))
        else:rejected.append(r)
    write_lines(out/'accepted.jsonl',accepted);write_lines(out/'rejected.jsonl',rejected)
    write_lines(out/'sft_train/train.jsonl',[{'conversations':r['conversations']} for r in accepted])
    write_lines(out/'control_train/train.jsonl',[{'conversations':[
        {'from':'system','value':'你是数学助手。根据题意建立数量关系，仅输出一行“列式：数学表达式”。不要自己计算最终答案，结果由外部计算器执行。'},
        {'from':'human','value':r['question']},{'from':'gpt','value':'列式：'+r['expression']}]} for r in accepted])
    dump(out/'preview.readable.json',accepted[:20])
    summary=dict(accepted=len(accepted),rejected=len(rejected),processed=len(accepted)+len(rejected),source_count=len(rows),charged_or_reserved_tokens=client.state['charged_or_reserved_tokens'],budget=args.budget,stopped=client.halt,limitations='Teacher-generated, same-model separate review; not human ground truth. Arithmetic checks cannot prove semantic correctness. Local budget does not query remaining free quota.')
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
