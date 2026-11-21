"""Closed-loop multi-turn regression, same prompts and greedy decoding for each model."""
import argparse,hashlib,json,re,time
from collections import defaultdict
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from math_tutor.common import ROOT,read_records,write_lines,dump
from math_tutor.data.multiturn_data import SYSTEM

def cases():
    result=[]
    for i in range(6):
        a,b,c=4+i,7+i,11+i
        result.append({'id':f'change-{i}','kind':'condition_change','users':[f'青禾社有{a}盒书签，每盒{b}枚，总共有多少枚？',f'现在只把每盒数量改成{c}枚，盒数保持不变，重新计算。','只回答刚才修改后的总数，不要其他文字。'],'checks':[{'contains_number':a*b},{'contains_number':a*c},{'exact_number':a*c}]})
        result.append({'id':f'hint-{i}','kind':'hint','users':[f'松果社有{a}箱贴纸，每箱{c}张，共有多少张？先别告诉我答案，只给一个提示。','我该用加法还是乘法？仍然不要告诉我最后得数。','现在可以公布答案，请计算。'],'checks':[{'forbid_number':a*c},{'forbid_number':a*c,'contains_any':['乘','相加']},{'contains_number':a*c}]})
        result.append({'id':f'correct-{i}','kind':'correction','users':[f'云帆社有{a}袋玻璃珠，每袋{b}颗。我算成{a}+{b}={a+b}颗，这样对吗？','为什么不是把袋数加上每袋的数量？',f'我重新算成{a}×{b}={a*b}颗，对吗？'],'checks':[{'contains_any':['乘','相加','不对','不正确']},{'contains_any':['每','组','重复']},{'contains_any':['正确','对的','是的','没错','对了']}]})
        name=['小岚','阿澄','小棠','小栩','阿禾','小祈'][i]
        result.append({'id':f'chat-{i}','kind':'chat_memory','users':[f'你好，我叫{name}，我喜欢画画。请自然地和我打个招呼，不要列式或答案模板。','我今天学数学有点累，想先休息一下。','我刚才说我喜欢什么？只回答这个爱好。'],'checks':[{'forbid_any':['列式：','答案：']},{'forbid_any':['列式：','答案：']},{'contains_any':['画画','绘画']}]})
    return result

def check(text,spec):
    def number(n):return re.search(r'(?<![\d.])'+re.escape(str(n))+r'(?![\d.])',text) is not None
    outcomes=[]
    for key,value in spec.items():
        if key=='contains_number':outcomes.append(number(value))
        elif key=='forbid_number':outcomes.append(not number(value))
        elif key=='exact_number':outcomes.append(text.strip().rstrip('。')==str(value))
        elif key=='contains_any':outcomes.append(any(w in text for w in value))
        elif key=='forbid_any':outcomes.append(not any(w in text for w in value))
    return all(outcomes)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='models/qwen-0.5b');p.add_argument('--adapter');p.add_argument('--output',required=True);p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise SystemExit('Output exists; choose new output')
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(ROOT/args.model,padding_side='left');model=AutoModelForCausalLM.from_pretrained(ROOT/args.model,dtype=torch.float16,attn_implementation='sdpa').to('cuda')
    if args.adapter:model=PeftModel.from_pretrained(model,ROOT/args.adapter)
    model.eval();rows=[];scores=defaultdict(list);suite=cases();suite=suite[:args.limit] if args.limit else suite
    write_lines(out.with_suffix('.cases.jsonl'),suite)
    start=time.monotonic()
    for case in suite:
        messages=[{'role':'system','content':SYSTEM}];turns=[]
        for user,spec in zip(case['users'],case['checks']):
            messages.append({'role':'user','content':user});inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():ids=model.generate(**inputs,do_sample=False,max_new_tokens=args.max_new_tokens,pad_token_id=tok.eos_token_id)
            tail=ids[0,inputs.input_ids.shape[1]:];text=tok.decode(tail,skip_special_tokens=True).strip();passed=check(text,spec)
            turns.append({'user':user,'response':text,'check':spec,'heuristic_pass':passed,'output_tokens':len(tail),'hit_limit':len(tail)>=args.max_new_tokens})
            messages.append({'role':'assistant','content':text});scores[case['kind']].append(passed)
        rows.append({'id':case['id'],'kind':case['kind'],'turns':turns,'all_checks_pass':all(t['heuristic_pass'] for t in turns),'needs_human_review':True})
        print(case['id'],[t['heuristic_pass'] for t in turns],flush=True)
    write_lines(out,rows)
    dump(out.with_suffix('.summary.json'),{'model':args.model,'adapter':args.adapter,'episodes':len(rows),'turns':sum(len(r['turns']) for r in rows),'episode_all_checks_pass':sum(r['all_checks_pass'] for r in rows),'turn_heuristic_pass':{k:{'pass':sum(v),'total':len(v),'rate':sum(v)/len(v)} for k,v in scores.items()},'do_sample':False,'max_new_tokens':args.max_new_tokens,'system_prompt':SYSTEM,'suite_sha256':hashlib.sha256(out.with_suffix('.cases.jsonl').read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((ROOT/args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest() if args.adapter else None,'seconds':time.monotonic()-start,'limitations':'Closed-loop histories contain each model own outputs. Keyword/number heuristics only; not semantic or teaching quality scores. Small authored regression suite, no claim of blind generalization.'})
if __name__=='__main__':main()
