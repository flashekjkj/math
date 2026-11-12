"""OpenAI 兼容推理服务 + 内置网页演示。

启动：
python -m math_tutor serve
python -m math_tutor serve --model models/qwen-0.5b --adapter runs/local-sft-v1

接口：
  GET  /health               服务与显存状态
  GET  /v1/models            已加载模型
  POST /v1/chat/completions  OpenAI 格式，支持 stream:true
  GET  /api/prompts          网页使用的 prompts/ 提示词
  GET  /                     内置聊天页
"""
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from math_tutor.common import CHAT_SYSTEM, ROOT, SYSTEM
from math_tutor.inference.tutor_session import TUTOR_SYSTEM, finish_solution, route_turn
from .engine import ModelEngine
from .webui import PAGE


def load_prompts():
    prompts = {'chat': CHAT_SYSTEM}
    for name in ('solve', 'tutor'):
        path = ROOT / 'prompts' / f'{name}.txt'
        if path.exists():
            prompts[name] = path.read_text(encoding='utf-8-sig').strip()
    return prompts


def _messages_from(body):
    if not isinstance(body, dict):
        return None
    messages = body.get('messages')
    if not isinstance(messages, list) or not messages:
        return None
    for m in messages:
        if not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') \
                or not isinstance(m.get('content'), str):
            return None
    return messages


def create_app(args):
    engine = ModelEngine(args.model, args.adapter, args.max_new_tokens)
    prompts = load_prompts()
    app = FastAPI(title='Math Tutor', version='0.1.0')
    app.state.engine = engine

    @app.get('/health')
    def health():
        return engine.info()

    @app.get('/v1/models')
    def models():
        return dict(object='list', data=[dict(id='math-tutor', object='model',
                                              created=int(time.time()),
                                              owned_by='math-tutor-project')])

    @app.get('/api/prompts')
    def api_prompts():
        return prompts

    @app.post('/v1/chat/completions')
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(dict(error='请求体必须是 JSON'), status_code=400)
        messages = _messages_from(body)
        if messages is None:
            return JSONResponse(dict(error='messages 必须是 role/content 消息列表'), status_code=422)
        mode = body.get('mode')
        automatic = mode == 'auto' or (mode is None and messages[0]['role'] != 'system')
        turn = None
        if automatic:
            if messages[-1]['role'] != 'user' or not messages[-1]['content'].strip():
                return JSONResponse(dict(error='自动辅导需要以一条非空用户消息结束'), status_code=422)
            turn = route_turn(messages)
            if turn.reply is None:
                system = {'chat': CHAT_SYSTEM, 'solve': SYSTEM, 'tutor': TUTOR_SYSTEM}[turn.mode]
                messages = [dict(role='system', content=system),
                            *(m for m in messages if m['role'] != 'system')]
        elif mode is not None:
            if not isinstance(mode, str) or mode not in prompts:
                return JSONResponse(dict(error='未知对话模式'), status_code=422)
            if messages[0]['role'] != 'system':
                messages = [dict(role='system', content=prompts[mode]), *messages]
        use_adapter = turn.mode == 'solve' if turn else mode != 'chat'
        max_tokens = body.get('max_tokens') or engine.max_new_tokens
        temperature = float(body.get('temperature', 0.7))
        top_p = float(body.get('top_p', 0.9))
        want_stream = bool(body.get('stream', False))
        model_id = 'math-tutor'
        base = dict(id='chatcmpl-' + uuid.uuid4().hex[:16], object='chat.completion',
                    created=int(time.time()), model=model_id)

        def reply_chunks():
            if turn and turn.reply is not None:
                yield turn.reply
            elif turn and turn.mode == 'solve':
                raw = engine.generate(messages, max_tokens, temperature, top_p, use_adapter=use_adapter)
                yield finish_solution(raw)
            else:
                yield from engine.stream(messages, max_tokens, temperature, top_p, use_adapter=use_adapter)

        if not want_stream:
            try:
                content = ''.join(reply_chunks())
            except ValueError as e:
                return JSONResponse(dict(error=str(e)), status_code=413)
            deterministic = turn is not None and turn.reply is not None
            prompt_tokens = 0 if deterministic else engine._build_inputs(messages)['input_ids'].shape[1]
            completion_tokens = 0 if deterministic else len(engine.tokenizer.encode(content, add_special_tokens=False))
            return dict(**base,
                        choices=[dict(index=0, message=dict(role='assistant', content=content),
                                      finish_reason='stop')],
                        usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                   total_tokens=prompt_tokens + completion_tokens))

        def event_stream():
            try:
                first = True
                for piece in reply_chunks():
                    chunk = dict(**base, choices=[dict(index=0, delta=dict(role='assistant', content=piece),
                                                       finish_reason=None)])
                    yield ('data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n')
                    first = False
                if first:
                    yield 'data: ' + json.dumps(dict(**base, choices=[dict(
                        index=0, delta=dict(role='assistant', content=''), finish_reason=None)]),
                        ensure_ascii=False) + '\n\n'
                yield 'data: ' + json.dumps(dict(**base, choices=[dict(
                    index=0, delta=dict(), finish_reason='stop')]), ensure_ascii=False) + '\n\n'
                yield 'data: [DONE]\n\n'
            except ValueError as e:
                yield 'data: ' + json.dumps(dict(error=str(e)), ensure_ascii=False) + '\n\n'

        return StreamingResponse(event_stream(), media_type='text/event-stream')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return PAGE

    return app


def serve(args):
    import uvicorn
    app = create_app(args)
    info = app.state.engine.info()
    print(json.dumps(dict(模型已加载=info), ensure_ascii=False, indent=2))
    print('打开 http://%s:%d 使用网页演示' % (args.host, args.port))
    print('接口：POST http://%s:%d/v1/chat/completions（OpenAI 格式）' % (args.host, args.port))
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
