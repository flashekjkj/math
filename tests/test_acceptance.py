import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import httpx
from math_tutor.serve.acceptance import percentile, stream_request, evaluate, compare


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class AcceptanceTests(unittest.TestCase):
    def test_percentile(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(percentile([1, 2, 3], .95), 2.9)

    def test_failed_stream_is_not_counted_as_success(self):
        async def scenario():
            async with httpx.AsyncClient(base_url='http://mock', transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"8"}}]}\n\n'))) as client:
                result = await stream_request(client, 'q', Tokenizer(), 10)
                self.assertFalse(result['ok'])
                self.assertIn('DONE', result['error'])
        asyncio.run(scenario())

    def test_stream_success(self):
        async def scenario():
            async with httpx.AsyncClient(base_url='http://mock', transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, text='data: {"choices":[{"delta":{"content":"8"}}]}\n\ndata: [DONE]\n\n'))) as client:
                result = await stream_request(client, 'q', Tokenizer(), 10)
                self.assertTrue(result['ok'])
                self.assertEqual(result['output_tokens'], 1)
                self.assertIsNotNone(result['first_content_seconds'])
        asyncio.run(scenario())

    def test_request_failure_stays_in_accuracy_denominator(self):
        async def scenario(out):
            def handler(request):
                question=json.loads(request.content)['messages'][-1]['content']
                return httpx.Response(500) if question=='bad' else httpx.Response(200,
                    json={'choices':[{'message':{'content':'答案：8'}}]})
            async with httpx.AsyncClient(base_url='http://mock',transport=httpx.MockTransport(handler)) as client:
                _,result=await evaluate(client,[dict(id=1,question='ok',answer=8),dict(id=2,question='bad',answer=8)],out,10)
                self.assertEqual(result['numeric_accuracy'],.5)
                self.assertEqual(result['request_failures'],1)
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(scenario(Path(folder)))

    def test_comparison_rejects_different_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            left=Path(folder)/'left'; right=Path(folder)/'right'
            left.mkdir();right.mkdir()
            (left/'protocol.json').write_text(json.dumps({'data_sha256':'a'}))
            (right/'protocol.json').write_text(json.dumps({'data_sha256':'b'}))
            with self.assertRaisesRegex(ValueError,'data_sha256'):
                compare(left,right,Path(folder)/'comparison.json')
