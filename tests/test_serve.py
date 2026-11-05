import unittest
from math_tutor.serve.api import load_prompts
from math_tutor.serve.webui import PAGE


class ServeStaticTests(unittest.TestCase):
    def test_prompts_load(self):
        prompts = load_prompts()
        self.assertIn('solve', prompts)
        self.assertIn('tutor', prompts)
        self.assertTrue(prompts['solve'].startswith('你是'))

    def test_page_has_chat_ui(self):
        self.assertIn('v1/chat/completions', PAGE)
        self.assertIn('id="chat"', PAGE)
        self.assertIn('stream', PAGE)


if __name__ == '__main__':
    unittest.main()
