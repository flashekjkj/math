import unittest
from math_tutor.inference.calculator import execute
from math_tutor.data.build_relation_tool import relation

class CalculatorTests(unittest.TestCase):
    def test_valid(self):
        r=execute('关系：数量之和\n列式：2+3\n工具调用：{"name":"calculator","expression":"2+3"}')
        self.assertTrue(r['ok']); self.assertEqual(r['value'],5)
    def test_mismatch(self):
        self.assertFalse(execute('列式：2+3\n工具调用：{"name":"calculator","expression":"2+4"}')['ok'])
    def test_duplicate(self):
        self.assertFalse(execute('列式：2+3\n列式：2+4',False)['ok'])
    def test_unsafe_and_zero(self):
        for eq in ['__import__("os")','1/0','2**1000000']:
            self.assertFalse(execute('列式：'+eq,False)['ok'])
    def test_wrong_formula_not_repaired(self):
        self.assertEqual(execute('列式：2+4',False)['value'],6)
    def test_relation(self):
        self.assertIn('商',relation('45/(3/5)'))
    def test_preserve_analysis(self):
        self.assertTrue(execute('分析：先求单价，再求总价。\n列式：18/3*2',False)['response'].startswith('分析：'))
