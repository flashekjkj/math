"""Merge a causal-LM adapter in the same FP16 precision used for inference."""
import argparse
from math_tutor.common import ROOT,dump

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',default='models/qwen-0.5b')
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();out=ROOT/args.output
    if out.exists():raise SystemExit('Output exists; refusing overwrite')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    base=AutoModelForCausalLM.from_pretrained(str(ROOT/args.base_model),torch_dtype=torch.float16,device_map={'':0},trust_remote_code=False)
    model=PeftModel.from_pretrained(base,str(ROOT/args.adapter)).eval().merge_and_unload()
    model.config.use_cache=True
    model.save_pretrained(str(out),safe_serialization=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/args.base_model),trust_remote_code=False)
    tok.save_pretrained(str(out))
    dump(out/'merge_provenance.json',dict(base=args.base_model,adapter=args.adapter,dtype='float16'))
    print(out)

if __name__=='__main__':main()
