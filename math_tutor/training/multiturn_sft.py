"""Native chat-template SFT with exact assistant token masking and no truncation."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
from peft import LoraConfig,get_peft_model
from math_tutor.common import ROOT,read_records,dump

def encode_conversation(tok,conv,max_length=1024):
    messages=[{'role':{'system':'system','human':'user','gpt':'assistant'}[m['from']],'content':m['value']} for m in conv]
    ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False)
    if len(ids)>max_length:raise ValueError(f'Conversation exceeds {max_length}: {len(ids)}')
    labels=[-100]*len(ids)
    for i,m in enumerate(messages):
        if m['role']!='assistant':continue
        before=tok.apply_chat_template(messages[:i],tokenize=True,return_dict=False,add_generation_prompt=True)
        after=tok.apply_chat_template(messages[:i+1],tokenize=True,return_dict=False,add_generation_prompt=False)
        if before!=ids[:len(before)] or after!=ids[:len(after)]:raise ValueError('Non-prefix chat template')
        labels[len(before):len(after)]=ids[len(before):len(after)]
    assert any(t!=-100 for t in labels)
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/multiturn-sft-v1.json');args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'));out=ROOT/cfg['output_dir']
    if out.exists() and any(out.iterdir()):raise SystemExit('Choose a new empty output directory')
    manifest=json.loads((ROOT/'data/versions/multiturn-v1/manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('ready_for_training'):raise SystemExit('Dataset must pass final audit/export before training')
    set_seed(cfg['seed']);torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['model_name_or_path']);tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    ds={s:Dataset.from_list([encode_conversation(tok,r['conversations'],cfg['model_max_length']) for r in read_records(ROOT/f'data/versions/multiturn-v1/sft_{s}/{s}.jsonl')]) for s in ('train','valid')}
    out.mkdir(parents=True,exist_ok=True);dump(out/'run_config.json',cfg)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/args.config,Path(__file__),*list((ROOT/'data/versions/multiturn-v1').glob('sft_*/*.jsonl'))]}
    dump(out/'code_and_data_manifest.json',{'sha256':hashes,'train_rows':len(ds['train']),'valid_rows':len(ds['valid']),'base_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/cfg['model_name_or_path']).iterdir() if p.is_file()},'masking':'native template exact prefix, all assistant turns; no truncation'})
    (out/'environment.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    model=AutoModelForCausalLM.from_pretrained(ROOT/cfg['model_name_or_path'],dtype=torch.float16,attn_implementation='sdpa').to('cuda')
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['q_proj','v_proj']))
    for param in model.parameters():
        if param.requires_grad:param.data=param.data.float()
    model.enable_input_require_grads();model.config.use_cache=False
    training=TrainingArguments(output_dir=str(out),per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=cfg['learning_rate'],num_train_epochs=cfg['num_train_epochs'],warmup_steps=5,fp16=True,gradient_checkpointing=True,
        logging_steps=5,save_strategy='no',eval_strategy='no',report_to='none',dataloader_num_workers=0,seed=cfg['seed'],optim='adamw_torch')
    trainer=Trainer(model=model,args=training,train_dataset=ds['train'],eval_dataset=ds['valid'],processing_class=tok,data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100))
    result=trainer.train();trainer.save_model(str(out));tok.save_pretrained(out);trainer.save_state();trainer.save_metrics('train',result.metrics)
    metrics=trainer.evaluate();trainer.save_metrics('eval',metrics)
    b=[p.detach().float() for name,p in model.named_parameters() if 'lora_B' in name]
    dump(out/'completion.json',{'completed':True,'updates':trainer.state.global_step,'nonzero_lora_B':sum(bool(p.count_nonzero()) for p in b),'lora_B_norm':sum(float((p*p).sum()) for p in b)**0.5,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'adapter_sha256':hashlib.sha256((out/'adapter_model.safetensors').read_bytes()).hexdigest(),'metrics':metrics})
    print('SFT completed',result.metrics,metrics,flush=True)
if __name__=='__main__':main()
