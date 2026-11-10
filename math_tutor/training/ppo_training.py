"""Single-GPU PPO with a shared Qwen backbone and a trainable value head."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from math_tutor.common import ROOT, dump, read_records
from math_tutor.rewards.math_reward import correctness_reward


def advantages(rewards, values, gamma=1.0, lam=0.95):
    """A response is a finite episode. The terminal state has value zero."""
    result = torch.zeros_like(values)
    next_value = accumulator = 0.0
    for position in range(len(rewards) - 1, -1, -1):
        delta = rewards[position] + gamma * next_value - values[position]
        accumulator = delta + gamma * lam * accumulator
        result[position] = accumulator
        next_value = values[position]
    return result, result + values


def response_stats(policy, value_head, tokens, start, temperature):
    """At position t the model predicts token t+1 and values the prefix at t."""
    length = tokens.shape[1] - start
    outputs = policy(tokens, use_cache=False, logits_to_keep=length + 1,
                     output_hidden_states=value_head is not None)
    scores = outputs.logits[0, :-1].float() / temperature
    targets = tokens[0, start:]
    log_probs = scores.gather(1, targets[:, None]).squeeze(1) - scores.logsumexp(dim=-1)
    values = None
    if value_head is not None:
        hidden = outputs.hidden_states[-1][0, start - 1:-1]
        values = value_head(hidden.float()).squeeze(-1)
    return log_probs, values


def load_split(directory, limit):
    rows = [row for path in sorted(Path(directory).glob('*.jsonl')) for row in read_records(path)]
    if not rows:
        raise ValueError(f'No questions in {directory}')
    return rows[:limit]


def verify_start(model_path):
    dpo = json.loads((ROOT / 'runs/dpo-demo-v1/run_config.json').read_text(encoding='utf-8-sig'))
    grpo = json.loads((ROOT / 'runs/grpo-v1/base_model_manifest.json').read_text(encoding='utf-8-sig'))
    if any(Path(path).resolve() != model_path for path in (dpo['model_name_or_path'], grpo['model'])):
        raise ValueError('PPO must use the same SFT path as DPO and GRPO')
    hashes = {}
    for name, expected in grpo['sha256'].items():
        with (model_path / name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if hashes[name] != expected:
            raise ValueError(f'SFT file changed since GRPO: {name}')
    return {'model': str(model_path), 'sha256': hashes, 'dpo_path_verified': True,
            'grpo_sha256_verified': True}


@torch.no_grad()
def collect(policy, value_head, tokenizer, batch, gen_config, beta):
    prompts = [tokenizer.apply_chat_template(row['prompt'], tokenize=True,
               add_generation_prompt=True, return_dict=False) for row in batch]
    if max(map(len, prompts)) > 256:
        raise ValueError('Prompt exceeds the 256-token limit')
    inputs = tokenizer.pad({'input_ids': prompts}, padding=True, return_tensors='pt').to('cuda')
    width = inputs.input_ids.shape[1]
    output = policy.generate(**inputs, generation_config=gen_config)
    eos = gen_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    samples = []
    records = []
    for row, prompt, tail in zip(batch, prompts, output[:, width:]):
        ids = tail.tolist()
        stop = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
        ids = ids[:stop]
        tokens = torch.tensor([prompt + ids], device='cuda')
        start = len(prompt)
        answer = tokenizer.decode(ids, skip_special_tokens=True)
        reward, = correctness_reward([answer], [row['answer']], [row['question']])
        old, values = response_stats(policy, value_head, tokens, start, gen_config.temperature)
        with policy.disable_adapter():
            reference, _ = response_stats(policy, None, tokens, start, gen_config.temperature)
        sampled_kl = old - reference
        rewards = -beta * sampled_kl
        rewards[-1] += reward
        adv, returns = advantages(rewards, values)
        samples.append((tokens, start, old, values, adv, returns))
        records.append({'id': row['id'], 'completion': answer, 'reward': reward,
                        'response_tokens': len(ids), 'sampled_kl': sampled_kl.sum().item(),
                        'ended_with_eos': ids[-1] in eos_ids})
    return samples, records


def optimize(policy, value_head, samples, optimizer, scaler, temperature, clip, parameters):
    all_adv = torch.cat([sample[4] for sample in samples])
    mean, std = all_adv.mean(), all_adv.std(unbiased=False)
    total = len(all_adv)
    optimizer.zero_grad(set_to_none=True)
    metrics = dict(policy_loss=0.0, value_loss=0.0, clip_fraction=0.0)
    for tokens, start, old, old_values, adv, returns in samples:
        log_probs, values = response_stats(policy, value_head, tokens, start, temperature)
        adv = (adv - mean) / std.clamp_min(1e-8)
        ratio = (log_probs - old).exp()
        policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).sum() / total
        clipped_values = old_values + (values - old_values).clamp(-clip, clip)
        value_loss = 0.5 * torch.maximum((values - returns).square(),
                                          (clipped_values - returns).square()).sum() / total
        scaler.scale(policy_loss + 0.5 * value_loss).backward()
        metrics['policy_loss'] += policy_loss.item()
        metrics['value_loss'] += value_loss.item()
        metrics['clip_fraction'] += ((ratio - 1).abs() > clip).sum().item() / total
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return metrics | {'grad_norm': norm.item(), 'update_applied': scaler.get_scale() >= scale}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model_name_or_path', 'train_file_dir', 'validation_file_dir', 'output_dir'):
        parser.add_argument('--' + field, required=True)
    for field, default in {'max_train_samples': 200, 'max_eval_samples': 50, 'max_steps': 0,
                           'logging_steps': 5, 'batch_size': 4, 'ppo_epochs': 2,
                           'max_completion_length': 128, 'seed': 42}.items():
        parser.add_argument('--' + field, type=int, default=default)
    for field, default in {'learning_rate': 5e-6, 'value_learning_rate': 1e-4,
                           'temperature': 0.8, 'beta': 0.02, 'clip': 0.2}.items():
        parser.add_argument('--' + field, type=float, default=default)
    args = parser.parse_args()
    model_path = Path(args.model_name_or_path).resolve()
    manifest = verify_start(model_path)
    out = Path(args.output_dir)
    if (out / 'rollouts.jsonl').exists() or (out / 'completion.json').exists():
        raise ValueError('Training output already exists; choose a new output directory')
    dump(out / 'base_model_manifest.json', manifest)
    train = load_split(args.train_file_dir, args.max_train_samples)
    valid = load_split(args.validation_file_dir, args.max_eval_samples)
    if {row['family_id'] for row in train} & {row['family_id'] for row in valid}:
        raise ValueError('Train/validation family overlap')
    random.Random(args.seed).shuffle(train)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                 attn_implementation='sdpa')
    policy = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                                             lora_dropout=0.0, target_modules=['q_proj', 'v_proj']))
    policy.to('cuda').eval()
    value_head = torch.nn.Linear(model.config.hidden_size, 1, device='cuda')
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    actor = [p for p in policy.parameters() if p.requires_grad]
    parameters = actor + list(value_head.parameters())
    optimizer = torch.optim.AdamW([{'params': actor, 'lr': args.learning_rate},
                                  {'params': value_head.parameters(), 'lr': args.value_learning_rate}],
                                  weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    config = GenerationConfig(do_sample=True, top_p=1.0, top_k=0, temperature=args.temperature,
                              max_new_tokens=args.max_completion_length, use_cache=True,
                              eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id)
    step = applied = rollout_count = 0
    started = time.perf_counter()
    with (out / 'rollouts.jsonl').open('x', encoding='utf-8') as rollouts, \
         (out / 'updates.jsonl').open('x', encoding='utf-8') as updates:
        for offset in range(0, len(train), args.batch_size):
            samples, records = collect(policy, value_head, tokenizer, train[offset:offset + args.batch_size],
                                       config, args.beta)
            rollout_count += len(records)
            for record in records:
                rollouts.write(json.dumps(record, ensure_ascii=False) + '\n')
            rollouts.flush()
            for _ in range(args.ppo_epochs):
                metrics = optimize(policy, value_head, samples, optimizer, scaler,
                                   args.temperature, args.clip, parameters)
                step += 1
                applied += int(metrics['update_applied'])
                metrics |= {'step': step, 'rollouts': rollout_count}
                updates.write(json.dumps(metrics, allow_nan=False) + '\n')
                updates.flush()
                if step % args.logging_steps == 0:
                    print(metrics, flush=True)
                if step == args.max_steps:
                    break
            if step == args.max_steps:
                break
    if applied != step:
        raise RuntimeError(f'Only {applied}/{step} PPO updates succeeded')
    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(value_head.state_dict(), out / 'value_head.pt')
    dump(out / 'completion.json', {'completed': True, 'optimizer_steps': applied,
                                   'rollouts': rollout_count, 'train_questions': len(train),
                                   'seconds': time.perf_counter() - started,
                                   'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()})

if __name__ == '__main__':
    main()
