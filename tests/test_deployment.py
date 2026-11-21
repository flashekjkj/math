import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from math_tutor.common import ROOT


def load(name):
    spec = importlib.util.spec_from_file_location('deploy_' + name, ROOT / 'deploy/vllm' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_shell_files_use_lf_without_bom(self):
        for path in (ROOT / 'deploy/vllm').glob('*.sh'):
            data = path.read_bytes()
            self.assertNotIn(b'\r', data, str(path))
            self.assertTrue(data.startswith(b'#!/usr/bin/env bash\n'))

    def test_default_start_uses_windows_docker_without_wsl(self):
        module = load('finish_all')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'deploy/vllm').mkdir(parents=True)
            with patch.object(module, 'ROOT', root), patch.object(module.subprocess, 'run') as run:
                module.start_deployment()
            calls = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(calls), 4)
            self.assertTrue(calls[0][1].endswith('freeze.py'))
            self.assertEqual(calls[-1][-2:], ['up', '-d'])
            self.assertTrue(all(cmd[0] != 'wsl' for cmd in calls))

    def test_existing_lock_reconfigures_local_path_without_refresh(self):
        module = load('finish_all')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            deploy = root / 'deploy/vllm'; deploy.mkdir(parents=True)
            (deploy / 'deployment-lock.json').write_text('{}')
            with patch.object(module, 'ROOT', root), patch.object(module.subprocess, 'run') as run:
                module.start_deployment()
            self.assertEqual(run.call_args_list[0].args[0][-1], '--configure-local')

    def test_reconfigure_preserves_digest_and_requires_unchanged_model(self):
        module = load('freeze')
        with tempfile.TemporaryDirectory() as folder:
            here = Path(folder)
            (here / 'compose.yaml').write_bytes(b'services: {}\n')
            lock = dict(image='vllm/vllm-openai@sha256:fixed', compose_sha256=hashlib.sha256(
                (here / 'compose.yaml').read_bytes()).hexdigest())
            (here / 'deployment-lock.json').write_text(json.dumps(lock))
            (here / 'model-manifest.json').write_text('{}')
            with patch.object(module, 'HERE', here), patch.object(module, 'ROOT', here), patch.object(
                    module, 'model_manifest', return_value={}), patch('sys.argv', ['freeze.py', '--configure-local']):
                module.main()
            self.assertEqual(json.loads((here / 'deployment-lock.json').read_text()), lock)
            self.assertEqual((here / '.env').read_text(), 'VLLM_IMAGE='+lock['image']+'\nPROJECT_DIR='+here.as_posix()+'\n')
            with patch.object(module, 'HERE', here), patch.object(module, 'model_manifest', return_value={'changed': True}), patch(
                    'sys.argv', ['freeze.py', '--configure-local']):
                with self.assertRaises(SystemExit):
                    module.main()
