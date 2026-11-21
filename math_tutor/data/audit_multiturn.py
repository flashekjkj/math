"""Second context-state audit; independent request, same teacher, not human gold."""
import json,hashlib
from concurrent.futures import ThreadPoolExecutor
from math_tutor.common import ROOT,dump
from math_tutor.data.multiturn_data import Client

PROMPT='''逐条审查教学对话，重点逐轮维护当前有效条件。用户修改一个条件后，后续必须默认继承新条件，除非明确恢复旧条件；不能悄悄回退。第一轮不能只有“这道题”而没有题目。条件不全的题必须询问再解答。检查数值、数量关系、前后指代、用户只要提示时不能泄露最终答案。不要把合理的局部思路提示当成泄露。发现错误就拒绝，不确定也拒绝。回复JSON：{"items":[{"id":"原id","valid":true或false,"reason":"理由"},...]}，每个id都要给结果。'''
def main():
    source=ROOT/'data/versions/multiturn-v1';out=source/'context-audit';out.mkdir(exist_ok=True)
    client=Client(out,350000)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((source/'records').glob('*.json'))]
    rows=[r for r in rows if r['status']=='accepted' and r['kind']!='chat' and 'context_audit' not in r]
    batches=[rows[i:i+4] for i in range(0,len(rows),4)]
    def work(batch):
        name=hashlib.sha256('|'.join(r['id'] for r in batch).encode()).hexdigest()[:16];path=out/(name+'.json')
        if path.exists():return
        payload=[{'id':r['id'],'kind':r['kind'],'conversations':r['conversations'][1:]} for r in batch]
        try:
            result=client.call(PROMPT,payload,name,'audit');items=result['items']
            if {x['id'] for x in items}!={r['id'] for r in batch}:raise ValueError('Missing audit IDs')
            dump(path,result)
        except Exception as e:dump(path,{'error':type(e).__name__,'items':[{'id':r['id'],'valid':False,'reason':'audit_unconfirmed'} for r in batch]})
        print('Audited',name,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,batches))
    counts={'accepted':0,'rejected':0}
    for p in out.glob('*.json'):
        if p.name in ('usage.json','summary.txt.json'):continue
        for item in json.loads(p.read_text(encoding='utf-8'))['items']:
            record=source/'records'/(item['id']+'.json');r=json.loads(record.read_text(encoding='utf-8'));r['context_audit']=item
            if item['valid'] is not True:r['status']='rejected';counts['rejected']+=1
            else:counts['accepted']+=1
            dump(record,r)
    dump(out/'summary.txt.json',counts);print(counts,flush=True)
if __name__=='__main__':main()
