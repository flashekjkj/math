"""Re-generate preselected historical examples for README; not a benchmark."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from math_tutor.common import ROOT,SYSTEM,read_records,dump,write_lines
from math_tutor.evaluation.scoring import score_response
from math_tutor.inference.calculator import execute
from math_tutor.data.multiturn_data import SYSTEM as MULTI_SYSTEM

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/readme-showcase-20261120');args=parser.parse_args()
    out=ROOT/args.output
    if out.exists():raise SystemExit('Output exists')
    out.mkdir(parents=True)
    lookup={r['id']:r for s in ('valid','test') for r in read_records(ROOT/f'data/processed/{s}.jsonl')}
    mathids=['test:4961','test:5846','test:16758','test:13381','test:17849','test:6096']
    dpoids=['train:14320','train:6113','train:14715','train:8951','train:15587','train:3752']
    hint={'id':'hint-1','users':['松果社有5箱贴纸，每箱12张，共有多少张？先别告诉我答案，只给一个提示。','我该用加法还是乘法？仍然不要告诉我最后得数。','现在可以公布答案，请计算。']}
    dump(out/'selection.json',{'math_ids':mathids,'dpo_ids':dpoids,'multi':hint,'selection':'Retrospectively selected from historical gains / hint regression examples, before this re-run. All selected candidates and all rerun outputs retained. Not a random test or aggregate improvement claim.'})
    jobs=[('base','models/qwen-0.5b',None,mathids,SYSTEM,192),('sft','models/qwen-0.5b','runs/local-sft-v1',mathids,SYSTEM,192),('analysis_sft','models/math-analysis-0.5b',None,dpoids,(ROOT/'data/versions/teacher-analysis-v1/system_prompt.txt').read_text(encoding='utf-8').strip(),256),('analysis_dpo','models/math-analysis-0.5b','runs/analysis-dpo-v1',dpoids,(ROOT/'data/versions/teacher-analysis-v1/system_prompt.txt').read_text(encoding='utf-8').strip(),256),('multiturn_sft','models/qwen-0.5b','runs/multiturn-sft-v1',[],MULTI_SYSTEM,128)]
    torch.set_num_threads(4);rows=[];meta=[]
    for name,path,adapter,ids,system,max_tokens in jobs:
        tok=AutoTokenizer.from_pretrained(ROOT/path);model=AutoModelForCausalLM.from_pretrained(ROOT/path,dtype=torch.float16,attn_implementation='sdpa').to('cuda')
        if adapter:model=PeftModel.from_pretrained(model,ROOT/adapter)
        model.eval();meta.append({'name':name,'model':path,'adapter':adapter,'model_sha256':hashlib.sha256((ROOT/path/'model.safetensors').read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((ROOT/adapter/'adapter_model.safetensors').read_bytes()).hexdigest() if adapter else None})
        def generate(messages,limit):
            inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():output=model.generate(**inputs,do_sample=False,max_new_tokens=limit,pad_token_id=tok.eos_token_id)
            tail=output[0,inputs.input_ids.shape[1]:]
            return tok.decode(tail,skip_special_tokens=True).strip(),len(tail)>=limit
        for rid in ids:
            r=lookup[rid];text,trunc=generate([{'role':'system','content':system},{'role':'user','content':r['question']}],max_tokens)
            result={'model':name,'id':rid,'question':r['question'],'reference':r['answer'],'raw_response':text,'hit_limit':trunc,'system_prompt':system,'max_new_tokens':max_tokens}
            if name.startswith('analysis'):result['tool']=execute(text,require_call=False)
            else:result['numeric_score']=score_response(r['question'],text,r['answer'])
            rows.append(result);write_lines(out/'outputs.jsonl',rows);print(name,rid,json.dumps(result.get('numeric_score',{}).get('status',result.get('tool',{}).get('value'))),flush=True)
        if name in ('sft','multiturn_sft'):
            messages=[{'role':'system','content':MULTI_SYSTEM}];turns=[]
            for user in hint['users']:
                messages.append({'role':'user','content':user});text,trunc=generate(messages,128);messages.append({'role':'assistant','content':text});turns.append({'user':user,'raw_response':text,'hit_limit':trunc})
            rows.append({'model':name,'id':'hint-1','turns':turns,'system_prompt':MULTI_SYSTEM,'max_new_tokens':128});write_lines(out/'outputs.jsonl',rows)
        del model,tok;gc.collect();torch.cuda.empty_cache()
    dump(out/'protocol.json',{'models':meta,'do_sample':False,'batch_size':1,'device':'RTX3050 Laptop 4GB','selection':'Selected illustrative cases, not a benchmark. No retraining or checkpoint selection. Raw text unedited. Numeric scoring does not prove reasoning correctness. Analysis cases use deterministic calculator.','outputs_sha256':hashlib.sha256((out/'outputs.jsonl').read_bytes()).hexdigest()})
if __name__=='__main__':main()
