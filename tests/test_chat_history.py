import argparse
import copy
from unittest.mock import patch
import unittest
from math_tutor.inference.runtime import chat


class ChatHistoryTests(unittest.TestCase):
    def run_chat(self, automatic=False, keep_history=False):
        calls=[]
        args=argparse.Namespace(model='unused',adapter=None,tool_calculation=False,
             calculator_only=not automatic,solve=False,system_prompt_file=None,
             max_new_tokens=256,keep_history=keep_history)
        def generate(tokenizer,model,messages,length):
            calls.append(copy.deepcopy(messages))
            return '分析：计算当前算式。\n列式：52+65'
        with patch('math_tutor.inference.runtime.load_inference',return_value=(None,None)), \
             patch('math_tutor.inference.runtime.generate',side_effect=generate), \
             patch('builtins.input',side_effect=['给我出一道题','52+65','/exit']), \
             patch('builtins.print'):
            chat(args)
        return calls

    def test_fixed_solver_does_not_inherit_an_unrelated_turn(self):
        calls=self.run_chat()
        self.assertEqual(len(calls[1]),2)
        self.assertEqual(calls[1][-1]['content'],'52+65')

    def test_explicit_history_is_available(self):
        calls=self.run_chat(keep_history=True)
        self.assertEqual(len(calls[1]),4)
        self.assertEqual(calls[1][1]['content'],'给我出一道题')
