import unittest
from transformers import AutoTokenizer
from math_tutor.common import ROOT
from math_tutor.training.multiturn_sft import encode_conversation

class MultiturnMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen-0.5b',local_files_only=True)
        cls.conv=[{'from':'system','value':'你是助手。'},{'from':'human','value':'我喜欢画画。'},{'from':'gpt','value':'我记住了。'},{'from':'human','value':'我喜欢什么？'},{'from':'gpt','value':'你喜欢画画。'}]
    def test_native_ids_and_only_assistant_supervision(self):
        encoded=encode_conversation(self.tok,self.conv)
        messages=[{'role':{'system':'system','human':'user','gpt':'assistant'}[m['from']],'content':m['value']} for m in self.conv]
        expected=self.tok.apply_chat_template(messages,tokenize=True,return_dict=False)
        self.assertEqual(encoded['input_ids'],expected)
        supervised=self.tok.decode([t for t in encoded['labels'] if t!=-100])
        self.assertIn('我记住了',supervised)
        self.assertIn('你喜欢画画',supervised)
        self.assertNotIn('我喜欢什么',supervised)
        self.assertNotIn('你是助手',supervised)
        self.assertIn(self.tok.eos_token,supervised)
    def test_long_conversation_rejected_instead_of_losing_last_turn(self):
        with self.assertRaises(ValueError):
            encode_conversation(self.tok,self.conv,max_length=12)

if __name__=='__main__':unittest.main()
