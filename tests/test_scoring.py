import unittest
from math_tutor.evaluation.scoring import extract_details, score_response, summarize

class ScoringTests(unittest.TestCase):
    def test_bare_fraction(self):
        self.assertEqual(extract_details('1/10')['prediction'],.1)
    def test_percent(self):
        self.assertEqual(extract_details('答案：25%')['prediction'],.25)
    def test_latex_fraction(self):
        self.assertEqual(extract_details(r'答案：\frac{4}{5}')['prediction'],.8)
    def test_chinese_conclusion(self):
        self.assertEqual(extract_details('答案：数值\n所以，A是C的4/5倍。')['prediction'],.8)
    def test_short_answer(self):
        self.assertEqual(extract_details('答：平均每次运12吨。')['prediction'],12)
    def test_final_equation(self):
        self.assertEqual(extract_details(r'解得：\n\[ x = 60 \]'+'\n答案：数值')['prediction'],60)
    def test_intermediate_conclusion_is_not_final(self):
        r=score_response('多少千米？','所以速度=33.75千米/小时。\n所以路程=140千米。\n因此，路程为140千米。',100)
        self.assertEqual(r['status'],'incorrect')
        self.assertEqual(r['prediction'],140)
    def test_evaluate_handles_review_without_gpu(self):
        import argparse, tempfile, json
        from pathlib import Path
        from unittest.mock import patch
        import math_tutor.evaluation.runner as starter
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp)
            starter.write_lines(root/'data.jsonl',[
                dict(id='a',question='多少？',answer=5),
                dict(id='b',question='多少？',answer=5)])
            args=argparse.Namespace(output='out.jsonl',data='data.jsonl',limit=0,
                 model='fake',adapter=None,max_new_tokens=192)
            with patch.object(starter,'ROOT',root), patch.object(starter,'load_inference',return_value=(None,None)), patch.object(starter,'generate',side_effect=['答案：5','不知道']):
                starter.evaluate(args)
            summary=json.loads((root/'out.summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['numeric_accuracy'],.5)
            self.assertEqual(summary['status_counts']['unparsed'],1)
    def test_conflict(self):
        self.assertEqual(extract_details('答案：7500\n所以沙石重量为6000吨。')['status'],'conflict')
    def test_no_reference_guided_selection(self):
        response='答案：7500\n所以沙石重量为6000吨。'
        for gold in (7500,6000,123):
            self.assertEqual(score_response('多少吨？',response,gold)['status'],'conflict')
    def test_does_not_recompute(self):
        self.assertEqual(score_response('多少？','(900÷12)×30×12=8100',27000)['status'],'incorrect')
    def test_not_arbitrary_last_number(self):
        self.assertIsNone(extract_details('中间算出了5，然后加3')['prediction'])
    def test_multiple_answers(self):
        self.assertEqual(extract_details('答案：2或3')['status'],'unparsed')
    def test_multi_question(self):
        self.assertEqual(score_response('多少立方厘米？面积是多少？','答案：80',80)['status'],'needs_review')
    def test_unparsed_not_incorrect(self):
        self.assertIsNone(score_response('多少？','答案：数值',5)['correct'])
    def test_summary_keeps_total(self):
        rows=[score_response('多少？','答案：5',5),score_response('多少？','不知道',5)]
        s=summarize(rows)
        self.assertEqual(s['numeric_accuracy'],.5)
        self.assertEqual(s['coverage'],.5)
        self.assertEqual(s['judged_numeric_accuracy'],1)
    def test_thousands(self):
        self.assertEqual(extract_details('答案：27,000只')['prediction'],27000)

if __name__=='__main__': unittest.main()
