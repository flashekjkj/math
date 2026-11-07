"""Project-local training launcher. No sibling repository or Git command required."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from math_tutor.common import ROOT, dump

STAGES = {
    'sft': 'supervised_finetuning', 'dpo': 'dpo_training', 'pt': 'pretraining',
    'orpo': 'orpo_training', 'grpo': 'grpo_training', 'rm': 'reward_modeling',
    'ppo': 'ppo_training', 'opd': 'opd_training',
}

def launch(args):
    stage = getattr(args, 'stage', 'sft')
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8-sig'))
    for key in ('model_name_or_path', 'train_file_dir', 'validation_file_dir', 'output_dir', 'peft_path'):
        if config.get(key):
            config[key] = str((ROOT / config[key]).resolve())
    if args.smoke:
        config.update(max_train_samples=32, max_eval_samples=8, max_steps=5,
                      logging_steps=1, eval_steps=5, save_steps=5)
        config['output_dir'] += '-smoke'
    out = Path(config['output_dir'])
    module = 'math_tutor.training.' + STAGES[stage]
    command = [sys.executable, '-m', module]
    for key, value in config.items():
        command.extend(['--' + key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    print('本项目训练模块:', module, flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if getattr(args, 'dry_run', False):
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit('输出目录非空，请修改 output_dir；旧实验不会覆盖。')
    out.mkdir(parents=True, exist_ok=True)
    code_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT / 'math_tutor').rglob('*.py')}
    data_hashes = {}
    for key in ('train_file_dir', 'validation_file_dir'):
        if config.get(key):
            for p in Path(config[key]).rglob('*.jsonl'):
                data_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out / 'run_config.json', config)
    dump(out / 'code_and_data_manifest.json', {'stage': stage, 'module': module,
                                              'code_sha256': code_hashes, 'data_sha256': data_hashes})
    (out / 'environment.lock.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/datasets"))
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    env.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    subprocess.run(command, cwd=ROOT, check=True, env=env)
