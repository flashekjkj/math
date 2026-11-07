"""Project GRPO: Math23K outcome rewards on the same SFT checkpoint as DPO.
Original MedicalGPT implementation is retained in archive/grpo-before-20261106.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from math_tutor.common import ROOT, read_records, dump
from math_tutor.rewards.math_reward import correctness_reward, completion_text

@dataclass
class ProjectArguments:
    model_name_or_path: str = 'models/math-sft-0.5b'
    train_file_dir: str = 'data/versions/rl-v1/train'
    validation_file_dir: str = 'data/versions/rl-v1/valid'
    max_train_samples: int = 200
    max_eval_samples: int = 50
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: str = 'q_proj,v_proj'


def load_rows(directory, limit):
    rows = [r for p in sorted(Path(directory).glob('*.jsonl')) for r in read_records(p)]
    if not rows:
        raise ValueError('Empty dataset: ' + directory)
    return rows[:limit] if limit > 0 else rows


def main():
    args, cfg = HfArgumentParser((ProjectArguments, GRPOConfig)).parse_args_into_dataclasses()
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    model_path = Path(args.model_name_or_path).resolve()
    dpo = json.loads((ROOT/'runs/dpo-demo-v1/run_config.json').read_text(encoding='utf-8-sig'))
    if model_path != Path(dpo['model_name_or_path']).resolve():
        raise ValueError('GRPO must start from the same SFT model as DPO')
    files = sorted(p for p in model_path.iterdir() if p.is_file())
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(out/'base_model_manifest.json', {'model':str(model_path),'sha256':hashes,'dpo_run':'runs/dpo-demo-v1'})
    train = load_rows(args.train_file_dir, args.max_train_samples)
    valid = load_rows(args.validation_file_dir, args.max_eval_samples)
    if {r['family_id'] for r in train} & {r['family_id'] for r in valid}:
        raise ValueError('Train/valid overlap')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for row in train + valid:
        ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(ids) > 256:
            raise ValueError('Prompt exceeds audited 256-token limit')
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                attn_implementation='sdpa', trust_remote_code=False)
    peft = LoraConfig(task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha,
                      lora_dropout=0.0, target_modules=args.target_modules.split(','))

    def logged_reward(completions, answer, question, id, **kwargs):
        rewards = correctness_reward(completions, answer, question)
        with (out/'rollouts.jsonl').open('a', encoding='utf-8') as f:
            for uid, q, a, c, reward in zip(id, question, answer, completions, rewards):
                f.write(json.dumps({'id':uid,'question':q,'answer':a,'completion':completion_text(c),
                                    'reward':reward}, ensure_ascii=False)+'\n')
        return rewards

    trainer = GRPOTrainer(model=model, args=cfg, processing_class=tokenizer,
                          reward_funcs=logged_reward, peft_config=peft,
                          train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid))
    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.save_state()
    trainer.save_metrics('train', result.metrics)
    dump(out/'completion.json', {'completed':True,'global_step':trainer.state.global_step,
                                 'train_questions':len(train),'valid_questions':len(valid),
                                 'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                 'reward':'numeric-v2 correctness only; no claim of verified reasoning'})

if __name__ == '__main__':
    main()
