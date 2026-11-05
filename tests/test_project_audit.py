import unittest
from math_tutor.evaluation.project_audit import compare

class AuditTests(unittest.TestCase):
    def row(self, key='a', correct=None, response='答案：1'):
        return dict(id=key, question='题目', reference=1, response=response, correct=correct,
                    status='correct' if correct else 'unparsed')

    def test_unknown_to_correct_is_not_called_wrong_to_right(self):
        stats, _ = compare([self.row()], [self.row(correct=True)])
        self.assertEqual(stats['gained_confirmed_correct'], 1)
        self.assertEqual(stats['accuracy_delta_pp'], 100)

    def test_missing_ids_rejected(self):
        with self.assertRaises(ValueError): compare([self.row()], [self.row('b')])

    def test_reference_change_rejected(self):
        changed = self.row(); changed['reference'] = 2
        with self.assertRaises(ValueError): compare([self.row()], [changed])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): compare([self.row(), self.row()], [self.row()])
