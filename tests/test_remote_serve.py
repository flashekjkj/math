import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from math_tutor.serve.api import create_app
from math_tutor.serve.remote import RemoteModelEngine


class RemoteServeTests(unittest.TestCase):
    def client(self, handler):
        engine = RemoteModelEngine('http://upstream/v1', 'math-sft', 128,
            chat_model='base', transport=httpx.MockTransport(handler))
        args = SimpleNamespace(backend='vllm', adapter=None, base_url='http://upstream/v1',
            remote_model='math-sft', remote_chat_model='base', max_new_tokens=128, request_timeout=5)
        with patch('math_tutor.serve.api.RemoteModelEngine', return_value=engine), patch(
                'math_tutor.serve.api.ModelEngine', side_effect=AssertionError('loaded local weights')):
            return TestClient(create_app(args))

    def body(self, **kwargs):
        return dict(messages=[dict(role='user', content='题目')], mode='solve', **kwargs)

    def test_nonstream_and_model_routing(self):
        models = []
        def handler(request):
            models.append(json.loads(request.content)['model'])
            return httpx.Response(200, json={'choices': [{'message': {'content': '答案：8'}}]})
        with self.client(handler) as client:
            for mode in ('solve', 'chat'):
                body = self.body(); body['mode'] = mode
                response = client.post('/v1/chat/completions', json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['choices'][0]['message']['content'], '答案：8')
                self.assertNotIn('usage', response.json())
        self.assertEqual(models, ['math-sft', 'base'])

    def test_stream_and_truncation(self):
        for done in (True, False):
            text = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
            if done:
                text += 'data: {"choices":[],"usage":{}}\n\ndata: [DONE]\n\n'
            with self.client(lambda r: httpx.Response(200, text=text)) as client:
                response = client.post('/v1/chat/completions', json=self.body(stream=True))
                self.assertIn('你好', response.text)
                self.assertEqual('data: [DONE]' in response.text, done)
                self.assertEqual('"error"' in response.text, not done)

    def test_error_codes(self):
        for status, expected in ((400,422),(413,413),(429,429),(500,502)):
            with self.client(lambda r: httpx.Response(status)) as client:
                self.assertEqual(client.post('/v1/chat/completions', json=self.body()).status_code, expected)

    def test_timeout_and_health(self):
        def handler(request):
            raise httpx.ReadTimeout('test')
        with self.client(handler) as client:
            self.assertEqual(client.get('/health').json()['status'], 'configured')
            self.assertEqual(client.get('/ready').status_code, 503)
            self.assertEqual(client.post('/v1/chat/completions', json=self.body()).status_code, 504)

    def test_readiness_validates_model_names(self):
        for ids, status in ((['base','math-sft'],200), ([],503)):
            with self.client(lambda r: httpx.Response(200, json={'data':[{'id':i} for i in ids]})) as client:
                self.assertEqual(client.get('/ready').status_code, status)

    def test_parameter_validation(self):
        with self.client(lambda r: self.fail('must not request upstream')) as client:
            for value in (0,-1,True,'bad'):
                self.assertEqual(client.post('/v1/chat/completions', json=self.body(max_tokens=value)).status_code,422)

    def test_auto_solve_uses_postprocessor(self):
        with self.client(lambda r: httpx.Response(200,json={'choices':[{'message':{'content':'列式：3+5'}}]})) as client:
            with patch('math_tutor.serve.api.route_turn',return_value=SimpleNamespace(mode='solve',reply=None)), patch(
                    'math_tutor.serve.api.finish_solution',return_value='答案：8') as finish:
                body=self.body(); body['mode']='auto'
                response=client.post('/v1/chat/completions',json=body)
                self.assertEqual(response.json()['choices'][0]['message']['content'],'答案：8')
                finish.assert_called_once_with('列式：3+5')

    def test_concurrent_requests(self):
        async def scenario():
            active=peak=0
            async def handler(request):
                nonlocal active,peak
                active+=1; peak=max(peak,active)
                await asyncio.sleep(.01)
                active-=1
                return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}]})
            engine=RemoteModelEngine('http://upstream/v1','math-sft',transport=httpx.MockTransport(handler))
            self.assertEqual(await asyncio.gather(*(engine.agenerate([]) for _ in range(4))),['ok']*4)
            self.assertEqual(peak,4)
        asyncio.run(scenario())

    def test_closing_stream_closes_upstream(self):
        class Stream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'
                yield b'data: [DONE]\n\n'

            async def aclose(self):
                self.closed = True

        async def scenario():
            stream = Stream()
            engine = RemoteModelEngine('http://upstream/v1', 'math-sft',
                transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
            response = engine.astream([])
            self.assertEqual(await anext(response), 'one')
            await response.aclose()
            self.assertTrue(stream.closed)
        asyncio.run(scenario())

    def test_existing_local_backend_still_works(self):
        class FakeEngine:
            max_new_tokens = 128

            def stream(self, *args, **kwargs):
                yield 'local'

        args = SimpleNamespace(model='fake', adapter=None, max_new_tokens=128)
        with patch('math_tutor.serve.api.ModelEngine', return_value=FakeEngine()):
            with TestClient(create_app(args)) as client:
                response = client.post('/v1/chat/completions', json=self.body(stream=True))
                self.assertIn('local', response.text)
                self.assertIn('data: [DONE]', response.text)

