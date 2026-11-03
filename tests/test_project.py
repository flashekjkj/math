import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from math_tutor.common import write_lines
from math_tutor.data import prepare as data_module
from math_tutor.training import launcher

class ProjectTests(unittest.TestCase):
    def test_token_length_uses_input_ids_not_dictionary_keys(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                size = 7 if messages[0]['content']=='long' else 3
                return list(range(size)) if kwargs.get('return_dict') is False else {'input_ids':list(range(size)),'attention_mask':[1]*size}
        fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            for split in ['train','valid']:
                write_lines(root/f'data/processed/{split}.jsonl',[
                    {'conversations':[{'from':'human','value':'long'}]},
                    {'conversations':[{'from':'human','value':'short'}]}])
            with patch.dict(sys.modules,{'transformers':fake}),patch.object(data_module,'ROOT',root):
                data_module.audit(argparse.Namespace(model='fake',max_length=5,data_root='data'))
            report=json.loads((root/'data/length_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['train']['longest'],7)
            self.assertEqual(report['train']['after'],1)
    def test_launcher_uses_only_local_module(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            c=dict(model_name_or_path='models/a',train_file_dir='data/train',validation_file_dir='data/valid',output_dir='runs/new')
            (root/'configs/a.json').write_text(json.dumps(c))
            with patch.object(launcher,'ROOT',root),patch.object(launcher.subprocess,'check_output',return_value='fake-freeze'),patch.object(launcher.subprocess,'run') as run:
                launcher.launch(argparse.Namespace(stage='sft',config='configs/a.json',smoke=False,dry_run=False))
            command=run.call_args.args[0]
            self.assertEqual(command[1:3],['-m','math_tutor.training.supervised_finetuning'])
            self.assertNotIn('MedicalGPT',' '.join(command))
            self.assertTrue((root/'runs/new/code_and_data_manifest.json').exists())

if __name__=='__main__': unittest.main()
