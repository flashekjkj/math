import unittest
from math_tutor.rewards.math_reward import correctness_reward

class RewardTests(unittest.TestCase):
    def test_correct_wrong_conflict_and_unparsed(self):
        texts = ['列式：2+3\n答案：5', '答案：6', '答案：5\n答案：6', '有2个，又有3个', '']
        self.assertEqual(correctness_reward(texts, [5]*5, ['一共多少个？']*5), [1,0,0,0,0])

    def test_conversation_and_percent(self):
        self.assertEqual(correctness_reward([[{'role':'assistant','content':'答案：25%'}]], [.25], ['比例是多少？']), [1])

    def test_multi_question_is_not_rewarded(self):
        self.assertEqual(correctness_reward(['答案：5'], [5], ['有多少个，还要多少个？']), [0])

    def test_mismatched_batch_rejected(self):
        with self.assertRaises(ValueError):
            correctness_reward(['答案：5'], [], [])
