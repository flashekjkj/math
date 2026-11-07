import unittest
from math_tutor.evaluation.sft_diagnostics import inspect_response

class DiagnosticsTests(unittest.TestCase):
    def test_fraction_arithmetic_error(self):
        r=inspect_response('列式：16/3/(4/15)\n答案：80')
        self.assertEqual(r['arithmetic_status'],'inconsistent')
        self.assertEqual(r['equation_value'],20)
    def test_percent(self):
        self.assertEqual(inspect_response('列式：50*98%\n答案：49')['arithmetic_status'],'consistent')
    def test_conflict_not_last_answer_wins(self):
        self.assertEqual(inspect_response('列式：1+1\n答案：2\n列式：1+2\n答案：4')['arithmetic_status'],'needs_review')
    def test_unsupported_not_executed(self):
        self.assertEqual(inspect_response("列式：__import__('os').system('whoami')\n答案：1")['arithmetic_status'],'uncheckable')
    def test_rounding_is_not_exact(self):
        self.assertEqual(inspect_response('列式：5/3\n答案：1.67')['arithmetic_status'],'inconsistent')
