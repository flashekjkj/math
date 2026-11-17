"""Async OpenAI-compatible backend. No local weights, tokenizer or GPU required."""
import json
import os
import httpx


class UpstreamError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


class RemoteModelEngine:
    def __init__(self, base_url, model, max_new_tokens=256, timeout=120,
                 chat_model=None, transport=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.chat_model = chat_model or model
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.transport = transport

    def _client(self):
        # Local inference must not be routed through Windows/system proxy.
        key = os.environ.get('MATH_TUTOR_API_KEY')
        headers = {'Authorization': 'Bearer ' + key} if key else {}
        return httpx.AsyncClient(base_url=self.base_url + '/', headers=headers,
                                 timeout=self.timeout, trust_env=False,
                                 transport=self.transport)

    def info(self):
        return dict(status='configured', backend='vllm', device='remote',
                    model=self.model, chat_model=self.chat_model,
                    adapter=None, max_new_tokens=self.max_new_tokens,
                    vram_allocated_gb=None, vram_total_gb=None,
                    note='Remote configuration only; /ready probes upstream. GPU memory is not measured here.')

    async def ready(self):
        async with self._client() as client:
            try:
                response = await client.get('models')
                self._check(response)
                ids = {row['id'] for row in response.json()['data']}
                missing = {self.model, self.chat_model} - ids
                if missing:
                    raise UpstreamError('上游未提供配置的模型：' + ', '.join(sorted(missing)))
                return dict(status='ok', backend='vllm', models=sorted(ids))
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise UpstreamError('无法读取上游模型列表') from exc

    @staticmethod
    def _check(response):
        if response.is_success:
            return
        status = response.status_code
        message = '上游模型请求失败（HTTP %d）' % status
        if status in (400, 413, 422):
            message += '，请检查上下文长度与生成参数'
        raise UpstreamError(message, status if status in (413, 429) else 422 if status in (400, 422) else 502)

    def _payload(self, messages, max_new_tokens, temperature, top_p, use_adapter, stream):
        return dict(model=self.model if use_adapter else self.chat_model,
                    messages=messages, max_tokens=max_new_tokens or self.max_new_tokens,
                    temperature=temperature, top_p=top_p, stream=stream)

    async def agenerate(self, messages, max_new_tokens=None, temperature=0.7,
                        top_p=0.9, *, use_adapter=True):
        async with self._client() as client:
            try:
                response = await client.post('chat/completions', json=self._payload(
                    messages, max_new_tokens, temperature, top_p, use_adapter, False))
                self._check(response)
                return response.json()['choices'][0]['message']['content'] or ''
            except httpx.TimeoutException as exc:
                raise UpstreamError('上游模型响应超时', 504) from exc
            except httpx.HTTPError as exc:
                raise UpstreamError('无法连接上游模型服务') from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise UpstreamError('上游返回了无效的对话响应') from exc

    async def astream(self, messages, max_new_tokens=None, temperature=0.7,
                      top_p=0.9, *, use_adapter=True):
        async with self._client() as client:
            try:
                async with client.stream('POST', 'chat/completions', json=self._payload(
                        messages, max_new_tokens, temperature, top_p, use_adapter, True)) as response:
                    self._check(response)
                    data = []
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            data.append(line[5:].lstrip())
                        elif not line and data:
                            event = '\n'.join(data)
                            data = []
                            if event == '[DONE]':
                                return
                            chunk = json.loads(event)
                            if 'error' in chunk:
                                raise UpstreamError('上游流式生成失败')
                            choices = chunk.get('choices', [])
                            if choices:
                                content = choices[0].get('delta', {}).get('content')
                                if content:
                                    yield content
                    # A truncated stream must not look like a successful completion.
                    raise UpstreamError('上游流式连接提前结束，未收到 [DONE]')
            except httpx.TimeoutException as exc:
                raise UpstreamError('上游模型响应超时', 504) from exc
            except httpx.HTTPError as exc:
                raise UpstreamError('上游模型连接中断') from exc
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                raise UpstreamError('上游返回了无效的流式响应') from exc
