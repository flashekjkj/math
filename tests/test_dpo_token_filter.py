import ast
from pathlib import Path
import unittest


class DpoLengthFilterTests(unittest.TestCase):
    def test_filter_counts_tokens_and_eos_not_characters(self):
        source = Path(__file__).resolve().parents[1]/'math_tutor/training/dpo_training.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        filters = [node for node in ast.walk(tree) if isinstance(node,ast.Lambda)
                   and 'tokenizer.encode' in ast.unparse(node)]
        self.assertEqual(len(filters),2)
        class Tokenizer:
            def encode(self,text,**kwargs):
                return list(range(3*len(text)))
        for node in filters:
            fn = eval(compile(ast.Expression(node),str(source),'eval'),
                      {'tokenizer':Tokenizer(),'full_max_length':7})
            self.assertTrue(fn({'prompt':'题','chosen':'答','rejected':'错'}))
            self.assertFalse(fn({'prompt':'题','chosen':'答案','rejected':'错'}))
