"""无需模型或 GPU 的关键数据处理测试：python -m unittest -v test_starter"""
import tempfile
from pathlib import Path
import unittest
from math_tutor.data.prepare import expression, convert, family
from math_tutor.common import read_records
from math_tutor.evaluation.runner import extract_answer


class DataTests(unittest.TestCase):
    def test_continuous_json(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            p = Path(tmp) / "a.json"
            p.write_text('{\n"id":1\n}\n{\n"id":2\n}', encoding="utf-8")
            self.assertEqual([x["id"] for x in read_records(p)], [1, 2])

    def test_percent_and_mixed_fraction(self):
        self.assertAlmostEqual(expression("x=200*120%*80%")[1], 192)
        self.assertAlmostEqual(expression("3(1/2)*2")[1], 7)

    def test_dangerous_expression_rejected(self):
        for value in ["__import__('os')", "2**10000", "1/0", "(1).__class__"]:
            with self.assertRaises((ValueError, ZeroDivisionError)):
                expression(value)

    def test_reject_inconsistent_answer(self):
        rows = [{"id": "a", "original_text": "小明有2个苹果，又买3个，共多少个？",
                 "equation": "x=2+3", "ans": "6"}]
        good, bad = convert(rows, "train")
        self.assertEqual(len(good), 0)
        self.assertEqual(len(bad), 1)

    def test_answer_parser_does_not_guess(self):
        self.assertEqual(extract_answer("列式：2+3\n答案：5个"), 5)
        self.assertEqual(extract_answer("答案：1/2"), 0.5)
        self.assertEqual(extract_answer("答案：20%"), 0.2)
        self.assertIsNone(extract_answer("中间算出了5，然后加3"))
        self.assertIsNone(extract_answer("答案：2或3"))

    def test_same_template_same_family(self):
        self.assertEqual(family("小明有2个苹果，又买3个，共多少个？"),
                         family("小明有4个苹果，又买5个，共多少个？"))


if __name__ == "__main__":
    unittest.main()
