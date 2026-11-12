import unittest
from math_tutor.data.check_preference_candidates import check_candidate
from math_tutor.data.review_preference_candidates import validate_review


class CandidateChecks(unittest.TestCase):
    def check(self, text, finish='eos'):
        return check_candidate('一共有多少个？', 5, {'text':text,'finish_reason':finish})

    def test_correct_equation_and_answer(self):
        result = self.check('列式：2+3\n答案：5')
        self.assertTrue(result['numeric']['correct'])
        self.assertTrue(result['expressions'][0]['matches_output'])
        self.assertEqual(result['semantic_status'], 'pending')

    def test_correct_number_does_not_hide_bad_equation(self):
        result = self.check('列式：2+2\n答案：5')
        self.assertTrue(result['numeric']['correct'])
        self.assertIn('expression_answer_mismatch', result['flags'])

    def test_conflict(self):
        self.assertIn('conflict', self.check('答案：5\n答案：6')['flags'])

    def test_truncation(self):
        self.assertIn('truncated', self.check('答案：5','length')['flags'])

    def test_unsupported_equation_is_pending(self):
        result = self.check('列式：2+3=5\n答案：5')
        self.assertIn('unsupported_expression', result['flags'])

    def test_teacher_cannot_omit_candidate(self):
        with self.assertRaises(ValueError):
            validate_review({'question_status':'valid','reference_status':'correct','candidates':[]}, ['a'])

    def test_teacher_must_explain_decision(self):
        with self.assertRaises(ValueError):
            validate_review({'question_status':'valid','reference_status':'correct', 'candidates':[
                {'candidate_id':'a','verdict':'correct','confidence':'high','error_type':'none','reason':''}]}, ['a'])
