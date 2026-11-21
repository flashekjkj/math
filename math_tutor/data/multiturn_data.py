"""Versioned teacher-generated conversations, separate review and token-safe export."""
import argparse,hashlib,json,os,random,threading,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from math_tutor.common import ROOT,read_records,write_lines,dump

SYSTEM='你是中文数学辅导助手。结合上下文自然、简洁地回答。闲聊时正常交流，不强套解题格式；解题时说明关键数量关系，给出列式和答案。用户只要提示时只给提示，不泄露答案；条件不足先询问。用户修改条件后按新条件推理。不编造条件，不迎合错误结论。'
GEN='''你是中文教学对话数据标注员。根据任务生成真实、紧凑的多轮对话，仅返回JSON：{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."},...]}。
必须恰好3组human/gpt交替，后两轮必须依赖前文或改变前文条件。每个助手回答20至85个汉字左右。不要输出思考过程、markdown代码块、system消息。数学解释必须符合题意，数字计算准确。不要写机械套话或每轮重复题目。提示阶段禁止公布答案。非数学闲聊禁止强套“分析/列式/答案”格式。首轮用户必须包含题目内容，不得仅写“这道题”。clarification任务首轮写出一个具体但缺少必要数字的题目。其他数学任务首轮完整复述原题，不得更改已知条件。任务给出参考答案时请独立核验，不合理则返回{"reject":true}。'''
REVIEW='''你是独立数据质检员。检查给定对话：数学题意、计算、修改条件后的答案、历史指代、只要提示时不泄露答案、自然闲聊、角色交替。逐项核验，错误或不确定就拒绝。仅返回JSON：{"valid":true或false,"reason":"简短理由"}。不要因为给出参考答案就默认正确。'''
KINDS=['hint','correction','condition_change','clarification','followup','switch']
TASKS={
'hint':'学生先请求一点提示且不要答案，接着尝试列式，最后请求完整解答。第一轮不可公布最终列式或答案。',
'correction':'学生给出一个具体错误列式，请指出错因；学生修改后再核对；最后追问关键数量关系。',
'condition_change':'学生先完整解题，再明确修改一个数字条件，最后追问新旧结果为何不同。',
'clarification':'第一轮故意缺少一个必要条件，助手必须询问；学生补充该条件，最后再追问解释。',
'followup':'学生先解题，然后使用“这里为什么这样算”等指代追问，最后要求更简单解释。',
'switch':'先解题，随后一句自然闲聊或学习鼓励，最后回到原题的具体数量关系。',
'chat':'进行自然中文交流，包含记住用户偏好、学习压力支持、学习计划或概念解释，并在后文引用前文信息；不使用固定解题格式。'}
class Client:
    def __init__(self,out,budget):
        self.out=out;self.budget=budget;self.key=os.environ['DASHSCOPE_API_KEY'];self.lock=threading.Lock();self.model='qwen3.8-flash';self.stop=False
        self.usage=json.loads((out/'usage.json').read_text(encoding='utf-8')) if (out/'usage.json').exists() else {'tokens':0,'requests':[]}
    def call(self,system,payload,rid,stage):
        for attempt in range(4):
            with self.lock:
                if self.stop or self.usage['tokens']>=self.budget:raise RuntimeError('Stopped or local token budget exhausted')
                model=self.model
                idx=len(self.usage['requests']);entry={'id':rid,'stage':stage,'model':model,'status':'pending'};self.usage['requests'].append(entry);dump(self.out/'usage.json',self.usage)
            body=json.dumps({'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],'max_tokens':2600 if stage=='generate' else (900 if stage=='audit' else 180),'temperature':0.65 if stage=='generate' else 0,'enable_thinking':False},ensure_ascii=False).encode()
            req=urllib.request.Request('https://maas.qianwenaiapi.com/compatible-mode/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
            except urllib.error.HTTPError as e:
                try:err=json.loads(e.read());code=str(err.get('error',{}).get('code',''));message=str(err.get('error',{}).get('message',''))
                except Exception:code='unknown';message=''
                with self.lock:
                    entry.update(status='http_error',http_status=e.code,code=code);dump(self.out/'usage.json',self.usage)
                quota=any(w in (code+' '+message).lower() for w in ['free quota','quota exhausted','quota has been exhausted','套餐','额度已','arrearage','insufficient_quota']) and not any(w in message.lower() for w in ['per minute','per second','rate limit'])
                if quota and model=='qwen3.8-flash':
                    with self.lock:self.model='qwen3.8-2.4t-a95b'
                    print('Primary quota exhausted; switching to authorized fallback',flush=True);continue
                if e.code==429 and not quota:time.sleep(3*(attempt+1));continue
                with self.lock:self.stop=True
                raise RuntimeError(f'API HTTP {e.code}, code={code}; stopped without recording response body') from None
            except Exception:
                with self.lock:entry['status']='transport_unknown';self.usage['tokens']+=5000;dump(self.out/'usage.json',self.usage)
                raise ValueError('transport_unknown; reserved 5000 tokens, sample not retried') from None
            with self.lock:
                entry.update(status='received',usage=result.get('usage'),returned_model=result.get('model'),request_id=result.get('id'))
                tokens=result.get('usage',{}).get('total_tokens')
                if not isinstance(tokens,int):self.stop=True;raise RuntimeError('Missing usage')
                self.usage['tokens']+=tokens;dump(self.out/'usage.json',self.usage)
            choice=result['choices'][0]
            if choice['finish_reason']!='stop':raise ValueError('truncated')
            text=choice['message']['content'].strip()
            if text.startswith('```'):
                if '\n' not in text:raise ValueError('invalid_code_fence')
                text=text.split('\n',1)[1].rsplit('```',1)[0]
            return json.loads(text)
        raise RuntimeError('Retry limit')

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--budget',type=int,default=1400000);args=p.parse_args()
    out=ROOT/'data/versions/multiturn-v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text(encoding='utf-8')).get('ready_for_training'):
        raise SystemExit('Dataset frozen; use a new version for further generation')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen-0.5b')
    source=read_records(ROOT/'data/versions/teacher-analysis-v1/accepted.jsonl')
    frozen={r['family_id'] for split in ('valid','test') for r in read_records(ROOT/f'data/processed/{split}.jsonl')}
    assert not frozen & {r['family_id'] for r in source}
    families=sorted({r['family_id'] for r in source});random.Random(9127).shuffle(families);held=set(families[:80])
    train=[r for r in source if r['family_id'] not in held];valid=[r for r in source if r['family_id'] in held]
    random.Random(42).shuffle(train);random.Random(43).shuffle(valid)
    jobs=[]
    for split,rows,nmulti,nchat in [('train',train,180,120),('valid',valid,24,16)]:
        for i in range(nmulti+nchat+120):
            kind=KINDS[i%6] if i<nmulti+80 else 'chat';row=rows[i%len(rows)]
            jobs.append({'id':f'{split}-{i:03d}','split':split,'kind':kind,'family_id':row['family_id'] if kind!='chat' else f'chat-{split}-{i}',
                         'task':TASKS[kind],'variation':i,'question':row['question'] if kind!='chat' else '',
                         'analysis':row['analysis'] if kind!='chat' else '', 'expression':row['expression'] if kind!='chat' else ''})
    random.Random(20261120).shuffle(jobs)
    client=Client(out,args.budget)
    def work(job):
        path=out/'records'/(job['id']+'.json')
        if path.exists() or client.stop:return
        try:
            obj=client.call(GEN,job,job['id'],'generate');conv=obj['conversations']
            if len(conv)!=6 or any(m['from']!=('human' if i%2==0 else 'gpt') or not isinstance(m['value'],str) or not m['value'].strip() for i,m in enumerate(conv)):raise ValueError('roles_or_turns')
            record={'id':job['id'],'split':job['split'],'kind':job['kind'],'family_id':job['family_id'],'conversations':[{'from':'system','value':SYSTEM}]+conv}
            messages=[{'role':{'system':'system','human':'user','gpt':'assistant'}[m['from']],'content':m['value']} for m in record['conversations']]
            length=len(tok.apply_chat_template(messages,tokenize=True,return_dict=False))
            if length>960:raise ValueError('over_token_limit')
            verdict=client.call(REVIEW,{'task':job['task'],'reference_question':job['question'],'reference_expression':job['expression'],'conversation':conv},job['id'],'review')
            record.update(review=verdict,tokens=length,status='accepted' if verdict.get('valid') is True else 'rejected');dump(path,record)
        except ValueError as e:dump(path,{'id':job['id'],'status':'rejected','reason':str(e)[:150]})
        except (KeyError,TypeError,IndexError):dump(path,{'id':job['id'],'status':'rejected','reason':'schema'})
        except RuntimeError as e:print(str(e),flush=True);return
        print(job['id'],json.loads(path.read_text(encoding='utf-8'))['status'],flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(work,jobs))
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'records').glob('*.json'))]
    from collections import Counter
    counts=Counter((r.get('split','unknown'),r.get('kind','unknown'),r['status']) for r in records)
    dump(out/'candidate_report.json',{'counts':{'|'.join(k):v for k,v in counts.items()},'source_sha256':hashlib.sha256((ROOT/'data/versions/teacher-analysis-v1/accepted.jsonl').read_bytes()).hexdigest(),'prompt_sha256':hashlib.sha256((SYSTEM+GEN+REVIEW).encode()).hexdigest(),'requested_models':['qwen3.8-flash','qwen3.8-2.4t-a95b'],'note':'Candidates only. Run audit_multiturn then export_multiturn before training.'})
    print(dict(counts),flush=True)
    print('Candidate generation complete; audit and final export are required before training.',flush=True)

if __name__=='__main__':main()
