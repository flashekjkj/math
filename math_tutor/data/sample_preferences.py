"""Sample real, unmodified SFT responses from the fixed training split."""
import argparse
import json
from pathlib import Path
import random

from math_tutor.common import ROOT, read_records, write_lines
from math_tutor.data.preference_io import atomic_json, run_lock, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/math-sft-0.5b')
    parser.add_argument('--output', default='data/versions/preference-sample-v1')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1 or args.candidates < 2:
        raise ValueError('Positive limit and at least two candidates required')
    model_path = (ROOT / args.model).resolve()
    output = ROOT / args.output
    source = ROOT / 'data/processed/train.jsonl'
    pool = read_records(source)
    frozen = {row['family_id'] for split in ('valid', 'test')
              for row in read_records(ROOT / f'data/processed/{split}.jsonl')}
    if any(row['family_id'] in frozen for row in pool):
        raise ValueError('Train-family leakage')
    random.Random(args.seed).shuffle(pool)
    rows = pool[:args.limit]
    if len(rows) != args.limit or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Insufficient or duplicate source questions')
    previous = json.loads((ROOT / 'runs/grpo-v1/base_model_manifest.json').read_text(encoding='utf-8-sig'))
    if Path(previous['model']).resolve() != model_path:
        raise ValueError('Expected the same SFT model as earlier DPO/GRPO experiments')
    model_hashes = {name: sha256(model_path / name) for name in previous['sha256']}
    if model_hashes != previous['sha256']:
        raise ValueError('SFT model files changed since GRPO')
    system = json.loads((ROOT / 'results/sft-test.summary.json').read_text(encoding='utf-8-sig'))['system_prompt']
    manifest = dict(schema_version=1, source_sha256=sha256(source), model=str(model_path),
                    model_sha256=model_hashes, selected_ids=[r['id'] for r in rows],
                    seed=args.seed, candidates=args.candidates, system_prompt=system,
                    generation=dict(temperature=0.8, top_p=0.95, top_k=0, max_new_tokens=192))
    with run_lock(output, 'sample'):
        manifest_path = output / 'sampling_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Source/model/config changed; choose a new output version')
        atomic_json(manifest_path, manifest)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this sampler')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                    device_map={'':0}, attn_implementation='sdpa', trust_remote_code=False).eval()
        config = GenerationConfig(do_sample=True, num_return_sequences=args.candidates,
                    temperature=0.8, top_p=0.95, top_k=0, max_new_tokens=192,
                    eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
        eos = config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        records = []
        for index, row in enumerate(rows):
            destination = output / 'samples' / f'{index:04d}.json'
            if destination.exists():
                saved = json.loads(destination.read_text(encoding='utf-8'))
                if saved['id'] != row['id'] or len(saved['candidates']) != args.candidates:
                    raise ValueError('Saved record mismatch')
                records.append(saved)
                continue
            messages = [{'role':'system','content':system}, {'role':'user','content':row['question']}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                         return_tensors='pt', return_dict=True).to(model.device)
            if inputs.input_ids.shape[1] > 256:
                raise ValueError('Prompt exceeds audited 256-token limit')
            seed = args.seed + index
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=config)
            candidates = []
            for candidate_index, tail in enumerate(generated[:, inputs.input_ids.shape[1]:]):
                ids = tail.tolist()
                stop = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
                ids = ids[:stop]
                candidates.append(dict(candidate_id=f'{row["id"]}:candidate:{candidate_index}',
                    text=tokenizer.decode(ids, skip_special_tokens=True), seed=seed,
                    batch_index=candidate_index, generated_tokens=len(ids),
                    finish_reason='eos' if ids[-1] in eos_ids else 'length'))
            saved = {k:row[k] for k in ('id','family_id','question','equation','answer')}
            saved.update(candidates=candidates)
            atomic_json(destination, saved)
            records.append(saved)
            print(f'{index+1}/{len(rows)} questions sampled', flush=True)
        write_lines(output / 'samples.jsonl', records)
        atomic_json(output / 'sampling_summary.json', dict(questions=len(records),
                    candidates=sum(len(r['candidates']) for r in records),
                    unique_responses=sum(len({c['text'] for c in r['candidates']}) for r in records),
                    truncated=sum(c['finish_reason']=='length' for r in records for c in r['candidates'])))


if __name__ == '__main__':
    main()
