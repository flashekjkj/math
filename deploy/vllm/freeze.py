"""Run from WSL: python3 deploy/vllm/freeze.py. Uses downloaded image only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / 'models' / 'math-sft-0.5b'


def command(*args):
    return subprocess.check_output(args, text=True, encoding='utf-8').strip()


def model_manifest():
    if not (MODEL / 'config.json').is_file():
        raise SystemExit('Missing model config: ' + str(MODEL))
    files = {}
    for path in sorted(MODEL.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(MODEL).parts:
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    sha.update(block)
            files[path.relative_to(MODEL).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha.hexdigest())
    return dict(model='models/math-sft-0.5b', files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='vllm/vllm-openai:latest')
    parser.add_argument('--refresh', action='store_true', help='Explicitly replace previous image lock')
    parser.add_argument('--model-only', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Check model/config hashes against the saved lock')
    args = parser.parse_args()
    manifest = model_manifest()
    manifest_path = HERE / 'model-manifest.json'
    if args.verify:
        record = json.loads((HERE / 'deployment-lock.json').read_text(encoding='utf-8'))
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest != saved:
            raise SystemExit('Model files changed since capture. Review before refreshing the lock.')
        config_sha = hashlib.sha256((HERE / 'compose.yaml').read_bytes()).hexdigest()
        if config_sha != record['compose_sha256']:
            raise SystemExit('Compose configuration changed since capture. Review before refreshing the lock.')
        expected_env = 'VLLM_IMAGE=' + record['image'] + '\nPROJECT_DIR=' + str(ROOT) + '\n'
        if (HERE / '.env').read_text(encoding='utf-8') != expected_env:
            raise SystemExit('Local .env does not match the saved image/path lock.')
        print('Model/config/image settings match the saved deployment lock.')
        return
    if args.model_only:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Model hashes captured: ' + str(manifest_path))
        return
    lock = HERE / 'deployment-lock.json'
    if lock.exists() and not args.refresh:
        raise SystemExit('Lock exists. Keep it, or use --refresh for an intentional new deployment.')
    image = json.loads(command('docker', 'image', 'inspect', args.image))[0]
    digests = [d for d in image.get('RepoDigests', []) if 'vllm-openai@sha256:' in d]
    if not digests:
        raise SystemExit('No repository digest found. Do not use an unpinned image.')
    gpu = command('docker', 'run', '--rm', '--pull=never', '--gpus', 'all',
                  '--entrypoint', 'nvidia-smi', image['Id'],
                  '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader')
    versions = command('docker', 'run', '--rm', '--pull=never', '--entrypoint', 'python3',
                       image['Id'], '-c',
                       'import torch,json,importlib.metadata as m; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,vllm=m.version("vllm"))))')
    record = dict(captured_at=datetime.now(timezone.utc).isoformat(), image=digests[0],
                  image_id=image['Id'], gpu=gpu, versions=json.loads(versions),
                  model_manifest='model-manifest.json',
                  compose_sha256=hashlib.sha256((HERE / 'compose.yaml').read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    lock.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    # WSL absolute path is captured on the machine actually running Docker.
    (HERE / '.env').write_text('VLLM_IMAGE=' + digests[0] + '\nPROJECT_DIR=' + str(ROOT) + '\n', encoding='utf-8')
    print('Pinned deployment saved: ' + str(lock))


if __name__ == '__main__':
    main()
