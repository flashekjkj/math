"""Measure an already-running gateway without stopping any GPU services."""
import argparse
import asyncio
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import httpx
from transformers import AutoTokenizer
from math_tutor.common import read_records
from math_tutor.serve.acceptance import performance,save


def sample_gpu(stop,samples):
    while not stop.is_set():
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
                '--format=csv,noheader,nounits'],text=True,timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            for line in raw.strip().splitlines():
                index,used,total=map(int,line.split(','))
                samples.append(dict(time=time.time(),gpu=index,used_mib=used,total_mib=total))
        except Exception as exc:
            samples.append(dict(time=time.time(),error=str(exc)))
        stop.wait(.25)


async def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/math-sft-0.5b'),local_files_only=True)
    rows=read_records(ROOT/'data/processed/valid.jsonl')[:12]
    stop=threading.Event();samples=[]
    worker=threading.Thread(target=sample_gpu,args=(stop,samples),daemon=True)
    async with httpx.AsyncClient(base_url=args.base_url,timeout=180,trust_env=False) as client:
        response=await client.get('/ready');response.raise_for_status()
        worker.start()
        try:
            results=await performance(client,rows,tokenizer,[1,2,4],12,192)
        finally:
            stop.set();worker.join(timeout=6)
            save(out/'gpu-samples.json',samples)
        save(out/'performance.json',results)
    valid=[r['used_mib'] for r in samples if r.get('gpu')==0]
    save(out/'summary.json',dict(captured_at=datetime.now(timezone.utc).isoformat(),
        base_url=args.base_url,memory_sampled_peak_mib=max(valid) if valid else None,
        sample_count=len(valid),memory_note='Whole GPU, includes display/other processes. Sampled peak, not exact instantaneous or per-process peak.',
        timing_note='First content is a TTFT proxy. TPOT estimates use (end-to-end latency - first content latency)/(retokenized output tokens - 1), not engine token timestamps.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--output',required=True)
    asyncio.run(main(parser.parse_args()))
