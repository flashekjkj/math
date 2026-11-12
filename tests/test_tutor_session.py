import unittest

from math_tutor.inference.tutor_session import reference_for, route_turn


class TutorSessionTests(unittest.TestCase):
    def ask(self, text, history=()):
        return route_turn([*history, {'role': 'user', 'content': text}])

    def test_chat_and_math_route_separately(self):
        for message in ('今天北京温度是多少？', '百分之几是什么意思', '讲个笑话'):
            self.assertEqual(self.ask(message).mode, 'chat')
        self.assertIn('答案：15', self.ask('3×5=?').reply)

    def test_answer_only_can_be_checked(self):
        reply = self.ask('3+5等于多少？我的答案是7，对吗？').reply
        self.assertIn('最终数值有误', reply)
        self.assertIn('3+5 = 8', reply)

    def test_wrong_formula_with_consistent_arithmetic(self):
        reply = self.ask('题目：有3盒铅笔，每盒5支，一共有多少支？ 列式：3+5 答案：8').reply
        self.assertIn('问题在列式', reply)
        self.assertIn('与所列算式的计算结果一致', reply)
        self.assertIn('3×5 = 15', reply)

    def test_steps_report_first_confirmed_error(self):
        reply = self.ask('题目：3*5+2=?\n步骤：3*5=14\n14+2=19\n答案：19').reply
        self.assertIn('第 1 行有计算错误', reply)
        self.assertNotIn('第 2 行有计算错误', reply)

    def test_unverified_word_problem_separates_arithmetic(self):
        reply = self.ask('题目：一道条件不全的购物题\n列式：3*5\n答案：16').reply
        self.assertIn('计算有误', reply)
        self.assertIn('列式是否符合题意仍待核验', reply)
        self.assertNotIn('最终数值有误', reply)

    def test_practice_history_supports_hint_wrong_answer_retry_and_next(self):
        request = '给我一道乘法题'
        exercise = self.ask(request)
        self.assertEqual(exercise.mode, 'practice')
        history = [{'role': 'user', 'content': request}, {'role': 'assistant', 'content': exercise.reply}]
        question = exercise.reply.splitlines()[0].removeprefix('练习题：')
        expected = reference_for(question).value
        hint = self.ask('给我一点提示', history)
        self.assertEqual(hint.mode, 'tutor')
        self.assertNotIn('答案：', hint.reply)
        wrong = self.ask('0', history)
        self.assertIn('最终数值有误', wrong.reply)
        history.extend([{'role': 'user', 'content': '0'}, {'role': 'assistant', 'content': wrong.reply}])
        self.assertEqual(self.ask('重做', history).reply, exercise.reply)
        self.assertIn('最终数值正确', self.ask(f'答案是{expected:g}', history).reply)
        next_question = self.ask('下一题', history).reply
        self.assertNotEqual(next_question, exercise.reply)
        self.assertIn('×', next_question.splitlines()[0])
        self.assertEqual(self.ask('讲个笑话', history).mode, 'chat')
