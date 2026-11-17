"""Fixed validation-set evaluation and HTTP service performance measurement."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import httpx
from math_tutor.common import ROOT, SYSTEM, read_records
from math_tutor.evaluation.scoring import score_response, summarize


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * fraction
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def payload(question, stream=False, max_tokens=192):
    return dict(model='math-tutor', mode='solve', messages=[
        dict(role='system', content=SYSTEM), dict(role='user', content=question)],
        stream=stream, temperature=0, top_p=1, max_tokens=max_tokens)


async def evaluate(client, rows, out, max_tokens):
    outputs = []
    # Append per-question evidence immediately; an interrupted run is not a complete summary.
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as log:
        for i, row in enumerate(rows):
            start = time.perf_counter()
            try:
                response = await client.post('/v1/chat/completions', json=payload(row['question'], max_tokens=max_tokens))
                response.raise_for_status()
                text = response.json()['choices'][0]['message']['content']
                score = score_response(row['question'], text, row['answer'])
                result = dict(id=row['id'], question=row['question'], reference=row['answer'], response=text, **score)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                result = dict(id=row['id'], question=row['question'], reference=row['answer'], response='',
                              correct=None, prediction=None, status='request_error', error=str(exc))
            result['seconds'] = time.perf_counter() - start
            outputs.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + '\n'); log.flush()
            if (i + 1) % 10 == 0:
                print('evaluation %d/%d' % (i + 1, len(rows)), flush=True)
    summary = summarize(outputs)
    summary['request_failures'] = sum(r['status'] == 'request_error' for r in outputs)
    save(out / 'evaluation.json', summary)
    return outputs, summary


async def stream_request(client, question, tokenizer, max_tokens):
    start = time.perf_counter()
    first = None
    pieces = []
    finished = False
    error = None
    try:
        async with client.stream('POST', '/v1/chat/completions', json=payload(question, True, max_tokens)) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    continue
                event = line[5:].strip()
                if event == '[DONE]':
                    finished = True
                    break
                chunk = json.loads(event)
                if 'error' in chunk:
                    raise ValueError(str(chunk['error']))
                choices = chunk.get('choices', [])
                text = choices[0].get('delta', {}).get('content') if choices else None
                if text:
                    if first is None:
                        first = time.perf_counter() - start
                    pieces.append(text)
            if not finished:
                raise ValueError('missing SSE DONE')
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start
    text = ''.join(pieces)
    return dict(ok=finished and error is None, seconds=elapsed, first_content_seconds=first,
                output_tokens=len(tokenizer.encode(text, add_special_tokens=False)), error=error)


async def performance(client, rows, tokenizer, concurrency, count, max_tokens):
    reports = []
    for level in concurrency:
        sem = asyncio.Semaphore(level)
        async def one(index):
            async with sem:
                return await stream_request(client, rows[index % len(rows)]['question'], tokenizer, max_tokens)
        start = time.perf_counter()
        attempts = await asyncio.gather(*(one(i) for i in range(count)))
        wall = time.perf_counter() - start
        successes = [r for r in attempts if r['ok']]
        latency = [r['seconds'] for r in successes]
        first = [r['first_content_seconds'] for r in successes if r['first_content_seconds'] is not None]
        reports.append(dict(concurrency=level, requests=count, failures=count-len(successes),
            wall_seconds=wall, success_requests_per_second=len(successes)/wall,
            output_tokens_per_second=sum(r['output_tokens'] for r in successes)/wall,
            latency_p50_seconds=percentile(latency,.5), latency_p95_seconds=percentile(latency,.95),
            first_content_p50_seconds=percentile(first,.5), first_content_p95_seconds=percentile(first,.95),
            samples=attempts))
        print('performance concurrency=%d successes=%d/%d' % (level,len(successes),count), flush=True)
    return reports


async def run(args):
    from transformers import AutoTokenizer
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / args.data
    rows = read_records(dataset)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or args.requests < 1 or any(n < 1 for n in args.concurrency):
        raise ValueError('Empty dataset or invalid request/concurrency count')
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/math-sft-0.5b'), local_files_only=True)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), label=args.label,
        data=args.data, data_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        ids=[r['id'] for r in rows], system_prompt=SYSTEM, max_tokens=args.max_tokens,
        temperature=0, top_p=1, concurrency=args.concurrency, requests=args.requests,
        measurement='HTTP gateway; first content chunk latency is not exact first-token latency; tokens retokenized from output text; no TPOT claimed',
        gpu_note='GPU measurements are unavailable unless separately captured; health only reports framework allocation.',
        model_manifest_sha256=hashlib.sha256((ROOT/'deploy/vllm/model-manifest.json').read_bytes()).hexdigest())
    save(out/'protocol.json',manifest)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip('/'), timeout=180, trust_env=False) as client:
        ready = await client.get('/ready'); ready.raise_for_status()
        save(out/'ready.json',ready.json())
        smoke = await client.post('/v1/chat/completions',json=payload('计算3+5',max_tokens=64))
        smoke.raise_for_status(); save(out/'smoke.json',smoke.json())
        warmup = await stream_request(client,rows[0]['question'],tokenizer,args.max_tokens)
        save(out/'warmup.json',warmup)
        if not warmup['ok']:
            raise RuntimeError('Streaming smoke failed: '+str(warmup['error']))
        _, scores = await evaluate(client,rows,out,args.max_tokens)
        perf = await performance(client,rows,tokenizer,args.concurrency,args.requests,args.max_tokens)
        save(out/'performance.json',perf)
        health = await client.get('/health'); health.raise_for_status(); save(out/'health.json',health.json())
    save(out/'completion.json',dict(status='complete',evaluation_request_failures=scores['request_failures'],
        performance_request_failures=sum(r['failures'] for r in perf),
        all_requests_succeeded=scores['request_failures']==0 and all(r['failures']==0 for r in perf)))
    return out


def compare(left, right, output):
    left,right,output=Path(left),Path(right),Path(output)
    a=json.loads((left/'protocol.json').read_text(encoding='utf-8'))
    b=json.loads((right/'protocol.json').read_text(encoding='utf-8'))
    for field in ('data_sha256','ids','system_prompt','max_tokens','temperature','top_p','model_manifest_sha256','concurrency','requests'):
        if a[field]!=b[field]:
            raise ValueError('Incomparable protocols: '+field)
    for folder in (left,right):
        if not (folder/'completion.json').is_file():
            raise ValueError('Incomplete run: '+str(folder))
    ra=read_records(left/'predictions.jsonl'); rb=read_records(right/'predictions.jsonl')
    if [r['id'] for r in ra] != a['ids'] or [r['id'] for r in rb] != b['ids']:
        raise ValueError('Prediction IDs do not match protocol')
    pairs=list(zip(ra,rb))
    gains=sum(x['correct'] is not True and y['correct'] is True for x,y in pairs)
    losses=sum(x['correct'] is True and y['correct'] is not True for x,y in pairs)
    report=dict(left=a['label'],right=b['label'],n=len(pairs),gains=gains,regressions=losses,
        changed_responses=sum(x['response']!=y['response'] for x,y in pairs),
        left_scores=json.loads((left/'evaluation.json').read_text(encoding='utf-8')),
        right_scores=json.loads((right/'evaluation.json').read_text(encoding='utf-8')),
        left_performance=json.loads((left/'performance.json').read_text(encoding='utf-8')),
        right_performance=json.loads((right/'performance.json').read_text(encoding='utf-8')),
        note='Single runs on validation data, no statistical significance or production capacity claimed; failure rates must accompany speed figures. Same model hashes do not guarantee identical kernels, precision implementation, or generation defaults.')
    if output.exists():
        raise FileExistsError(output)
    save(output,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8002')
    p.add_argument('--label',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data',default='data/processed/valid.jsonl')
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--max-tokens',type=int,default=192)
    p.add_argument('--requests',type=int,default=12)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4])
    args=p.parse_args()
    asyncio.run(run(args))


if __name__=='__main__':
    main()
