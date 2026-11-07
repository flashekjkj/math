import unittest
from math_tutor.evaluation.diagnose_sft import diagnose

class DiagnosticTests(unittest.TestCase):
    def test_correct_formula_bad_answer(self):
        r=diagnose('列式：(16/3)/(4/15)\n答案：80',20)
        self.assertFalse(r['arithmetic_consistent'])
        self.assertTrue(r['expression_matches_reference_value'])
    def test_wrong_formula_can_be_consistent(self):
        r=diagnose('列式：2+2\n答案：4',5)
        self.assertTrue(r['arithmetic_consistent'])
        self.assertFalse(r['expression_matches_reference_value'])
    def test_multiple_outputs_are_not_silently_selected(self):
        self.assertEqual(diagnose('列式：1+1\n答案：2\n答案：3',2)['status'],'multiple_outputs')
    def test_unsafe_expression_rejected(self):
        self.assertEqual(diagnose('列式：__import__("os")\n答案：2',2)['status'],'uncheckable')
    def test_percent(self):
        self.assertTrue(diagnose('列式：50*98%\n答案：49',49)['arithmetic_consistent'])
    def test_power_preserved(self):
        self.assertTrue(diagnose('列式：2**3\n答案：8',8)['arithmetic_consistent'])
    def test_rounding_requires_review(self):
        self.assertFalse(diagnose('列式：5/3\n答案：1.67',1/3)['arithmetic_consistent'])
