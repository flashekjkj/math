"""Run in Windows math environment: python deploy/vllm/finish_all.py.
Stages are resumable and never overwrite prior outputs. No training or dataset modification.
"""
import argparse
import asyncio
from datetime import datetime
import importlib.util
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from math_tutor.serve.acceptance import run,compare,save


def launch(port,remote,folder):
    stdout=(folder/('gateway-%d.stdout.log'%port)).open('a',encoding='utf-8')
    stderr=(folder/('gateway-%d.stderr.log'%port)).open('a',encoding='utf-8')
    cmd=[sys.executable,'-m','math_tutor','serve','--port',str(port),'--max-new-tokens','192']
    cmd+=['--backend','vllm','--remote-model','math-sft'] if remote else ['--model','models/math-sft-0.5b']
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    stdout.close();stderr.close()
    return proc


def wait_ready(proc,port):
    with httpx.Client(trust_env=False,timeout=3) as client:
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError('Gateway exited; inspect gateway stderr log')
            try:
                response=client.get('http://127.0.0.1:%d/ready'%port)
                if response.status_code==200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise TimeoutError('Gateway/upstream not ready in 240 seconds')


def require_free(port):
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError('Port %d is already in use. Existing user services were not stopped.'%port)


def measure(folder,label,port):
    args=argparse.Namespace(base_url='http://127.0.0.1:%d'%port,label=label,
        output=str(folder/label),data='data/processed/valid.jsonl',limit=100,
        max_tokens=192,requests=12,concurrency=[1,2,4])
    result=asyncio.run(run(args))
    completed=json.loads((result/'completion.json').read_text(encoding='utf-8'))
    if not completed['all_requests_succeeded']:
        raise RuntimeError('Acceptance had failed requests; inspect '+str(result))
    return result


def start_deployment(via_wsl=False, distro=None):
    deploy=ROOT/'deploy/vllm'
    if via_wsl:
        wsl=['wsl']+(['--distribution',distro] if distro else [])
        linux_root='/mnt/'+ROOT.drive[0].lower()+ROOT.as_posix()[2:]
        subprocess.run(wsl+['--exec','bash',linux_root+'/deploy/vllm/start.sh'],check=True)
        return
    # Docker Desktop still uses its WSL2 backend; Ubuntu shell invocation is unnecessary.
    if (deploy/'deployment-lock.json').exists():
        subprocess.run([sys.executable,str(deploy/'freeze.py'),'--configure-local'],check=True)
    else:
        subprocess.run([sys.executable,str(deploy/'freeze.py')],check=True)
    subprocess.run([sys.executable,str(deploy/'freeze.py'),'--verify'],check=True)
    compose=['docker','compose','--env-file',str(deploy/'.env'),'-f',str(deploy/'compose.yaml')]
    subprocess.run(compose+['config','--quiet'],check=True)
    subprocess.run(compose+['up','-d'],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--via-wsl',action='store_true',help='Optional: run Compose through Ubuntu instead of Windows Docker CLI')
    parser.add_argument('--distro',help='Distribution used only with --via-wsl')
    parser.add_argument('--resume',type=Path,help='Existing results folder with a completed baseline')
    args=parser.parse_args()
    if os.name!='nt' and not args.baseline_only:
        raise SystemExit('Run this orchestrator from Windows math environment.')
    folder=args.resume.resolve() if args.resume else ROOT/'reports'/'vllm-acceptance'/datetime.now().strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True,exist_ok=args.resume is not None)
    spec=importlib.util.spec_from_file_location('deployment_freeze',ROOT/'deploy/vllm/freeze.py')
    freeze=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)
    current=freeze.model_manifest()
    saved=json.loads((ROOT/'deploy/vllm/model-manifest.json').read_text(encoding='utf-8'))
    if current!=saved:
        raise RuntimeError('Model differs from saved model-manifest; review before acceptance.')
    save(folder/'run-status.json',dict(status='running',phase='preflight'))
    proc=None
    try:
        wsl=['wsl']+(['--distribution',args.distro] if args.distro else [])
        # Check control access before touching GPU. Baseline-only is usable without Docker/WSL.
        if not args.baseline_only:
            subprocess.run(['docker','info'],check=True,stdout=subprocess.DEVNULL)
            running=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
            if running:
                gpu_containers=[]
                for name in running:
                    info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
                    if info.get('HostConfig',{}).get('DeviceRequests'):
                        gpu_containers.append(name)
                if gpu_containers:
                    raise RuntimeError('Stop existing GPU containers before full acceptance: '+', '.join(gpu_containers))
        baseline_done=folder/'transformers'/'completion.json'
        if baseline_done.is_file() and not json.loads(baseline_done.read_text(encoding='utf-8'))['all_requests_succeeded']:
            raise RuntimeError('Saved baseline contains failures; use a fresh folder.')
        if not baseline_done.is_file():
            if (folder/'transformers').exists():
                raise RuntimeError('Previous baseline is incomplete. Use a fresh output folder.')
            require_free(8002)
            save(folder/'run-status.json',dict(status='running',phase='transformers'))
            proc=launch(8002,False,folder)
            wait_ready(proc,8002)
            measure(folder,'transformers',8002)
            proc.terminate();proc.wait(timeout=30);proc=None
        if not (folder/'baseline-environment.json').exists():
            metadata=dict(python=sys.version,platform=platform.platform(),
                versions={name:importlib.metadata.version(name) for name in ('torch','transformers','fastapi','httpx')})
            save(folder/'baseline-environment.json',metadata)
        if args.baseline_only:
            save(folder/'run-status.json',dict(status='baseline_complete',next='Run with --resume '+str(folder)))
            print('BASELINE_COMPLETE '+str(folder),flush=True)
            return
        # Use the Linux project path; Windows and WSL share the same E: files.
        drive=ROOT.drive[0].lower()
        linux_root='/mnt/'+drive+ROOT.as_posix()[2:]
        save(folder/'run-status.json',dict(status='running',phase='vllm_start'))
        start_deployment(args.via_wsl,args.distro)
        require_free(8000)
        proc=launch(8000,True,folder)
        wait_ready(proc,8000)
        measure(folder,'vllm',8000)
        report=compare(folder/'transformers',folder/'vllm',folder/'comparison.json')
        save(folder/'run-status.json',dict(status='complete',gateway_pid=proc.pid,
            webpage='http://127.0.0.1:8000',evidence=str(folder)))
        save(folder/'runtime-lock-copy.json',json.loads((ROOT/'deploy/vllm/deployment-lock.json').read_text(encoding='utf-8')))
        markdown=['# vLLM acceptance results','', 'Validation set only; numeric scoring is not proof of reasoning or teaching quality.','',
            '| Backend | Confirmed numeric accuracy | Request failures |','|---|---:|---:|']
        for name,key in [('Transformers','left_scores'),('vLLM','right_scores')]:
            scores=report[key]
            markdown.append('| %s | %.1f%% | %d |'%(name,100*scores['numeric_accuracy'],scores['request_failures']))
        markdown+=['','Per-request evidence and concurrency measurements are in comparison.json. No speedup claim without matched successful samples.',
            '', 'The vLLM container and gateway are left running. The owned Transformers gateway was stopped to release GPU memory.']
        (folder/'README.md').write_text('\n'.join(markdown),encoding='utf-8')
        print('COMPLETE '+str(folder),flush=True)
        proc=None # Keep only the final successful remote gateway running.
    except Exception as exc:
        save(folder/'run-status.json',dict(status='failed',error=str(exc)))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate();proc.wait(timeout=30)


if __name__=='__main__':
    main()
