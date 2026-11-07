"""GPU generation and calculator-system evaluation; preserve original outputs."""
import argparse
import hashlib
import json
import math
from collections import Counter
from math_tutor.common import ROOT,SYSTEM,read_records,write_lines,dump
from math_tutor.inference.runtime import load_inference,generate
from math_tutor.inference.calculator import TOOL_SYSTEM,execute

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/qwen-0.5b')
    p.add_argument('--adapter',default='runs/relation-tool-v1')
    p.add_argument('--no-adapter',action='store_true',help='For a merged standalone model')
    p.add_argument('--data',default='data/processed/test.jsonl')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--formula-only',action='store_true',help='Execute a single formula; no duplicate JSON call required')
    p.add_argument('--system-prompt-file')
    args=p.parse_args()
    if args.no_adapter: args.adapter=None
    dest=ROOT/args.output
    if dest.exists():raise SystemExit('Output exists')
    system=(ROOT/args.system_prompt_file).read_text(encoding='utf-8-sig').strip() if args.system_prompt_file else (SYSTEM if args.formula_only else TOOL_SYSTEM)
    rows=read_records(ROOT/args.data)
    if args.limit:rows=rows[:args.limit]
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    import torch
    tok,model=load_inference(args); results=[]
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    for start in range(0,len(rows),args.batch_size):
        batch=rows[start:start+args.batch_size]
        messages=[[{'role':'system','content':system},{'role':'user','content':row['question']}] for row in batch]
        inputs=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,padding=True,
            return_tensors='pt',return_dict=True).to(model.device)
        if inputs['input_ids'].shape[1]>2048:raise ValueError('Input too long')
        with torch.inference_mode():
            ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.pad_token_id)
        texts=tok.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        for row,raw in zip(batch,texts):
            tool=execute(raw,require_call=not args.formula_only)
            correct=tool['ok'] and math.isclose(tool['value'],row['answer'],rel_tol=1e-5,abs_tol=1e-5)
            results.append(dict(id=row['id'],question=row['question'],reference=row['answer'],tool=tool,correct=correct))
            print(f'{len(results)}/{len(rows)} executable={tool["ok"]} correct={correct}',flush=True)
    write_lines(dest,results)
    n=len(results);ok=sum(r['tool']['ok'] for r in results);correct=sum(r['correct'] for r in results)
    dump(dest.with_suffix('.summary.json'),dict(n=n,correct=correct,tool_system_accuracy=correct/n if n else None,
        executable=ok,execution_rate=ok/n if n else None,accuracy_among_executable=correct/ok if ok else None,
        model=args.model,adapter=args.adapter,system_prompt=system,max_new_tokens=args.max_new_tokens,do_sample=False,batch_size=args.batch_size,require_json_call=not args.formula_only,
        data_sha256=hashlib.sha256((ROOT/args.data).read_bytes()).hexdigest(),
        note='System result, not independent model arithmetic accuracy. Calculator never sees reference. Value matching is not proof of formulation or explanation correctness.'))

if __name__=='__main__':main()
