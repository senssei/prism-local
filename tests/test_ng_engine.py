import unittest
from foundry_ng.engine import format_prompt

class TestNgEngine(unittest.TestCase):
    def test_format_prompt_phi4(self):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        formatted = format_prompt(messages, template="phi4")
        self.assertIn("<|system|>\nYou are a helpful coding assistant.<|end|>", formatted)
        self.assertIn("<|user|>\nHello!<|end|>", formatted)
        self.assertTrue(formatted.endswith("<|assistant|>\n"))

    def test_format_prompt_chatml(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User prompt."}
        ]
        formatted = format_prompt(messages, template="chatml")
        self.assertIn("<|im_start|>system\nSystem prompt.<|im_end|>", formatted)
        self.assertIn("<|im_start|>user\nUser prompt.<|im_end|>", formatted)
        self.assertTrue(formatted.endswith("<|im_start|>assistant\n"))

if __name__ == "__main__":
    unittest.main()
