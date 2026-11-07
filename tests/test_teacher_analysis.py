import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from math_tutor.data.teacher_analysis import validate, Client

class TeacherAnalysisTests(unittest.TestCase):
    def test_valid(self):
        text='先用总价除以购买数量求出每本价格，再乘以新购买的本数，得到新购买这些本子所需的费用。'
        self.assertEqual(validate(dict(valid=True,analysis=text,expression='18/3*2'),dict(answer=12))[1],'18/3*2')
    def test_bad_answer(self):
        with self.assertRaises(ValueError):validate(dict(valid=True,analysis='根据题目中的数量关系，先确定每份对应的数量，再计算所求的总量。',expression='2+4'),dict(answer=5))
    def test_rejected(self):
        with self.assertRaises(ValueError):validate(dict(valid=False,reason='缺图'),dict(answer=5))
    def test_budget_prevents_network(self):
        with tempfile.TemporaryDirectory() as temp:
            c=Client(Path(temp),'https://example.invalid','test-secret','model',1)
            with patch('urllib.request.urlopen') as network:
                with self.assertRaises(RuntimeError):c.call('prompt',{},'a','generate')
                network.assert_not_called()
    def test_usage_and_secret_not_persisted(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(dict(usage={'total_tokens':100},choices=[dict(finish_reason='stop',message={'content':'{"valid":true}'})])).encode()
        with tempfile.TemporaryDirectory() as temp:
            c=Client(Path(temp),'https://example.invalid','test-secret','model',10000)
            with patch('urllib.request.urlopen',return_value=Response()):self.assertTrue(c.call('prompt',{},'a','generate')['valid'])
            self.assertEqual(c.state['charged_or_reserved_tokens'],100)
            self.assertNotIn('test-secret',c.state_path.read_text())
    def test_transport_reservation_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            c=Client(Path(temp),'https://example.invalid','test-secret','model',10000)
            with patch('urllib.request.urlopen',side_effect=TimeoutError):
                with self.assertRaises(RuntimeError):c.call('prompt',{},'a','generate')
            self.assertTrue(c.halt)
            self.assertGreater(c.state['charged_or_reserved_tokens'],0)
