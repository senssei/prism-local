import unittest

from prism.templates import detect_template, format_prompt

MSGS = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "Bye"},
]


class TestTemplates(unittest.TestCase):
    def test_detect_template(self):
        self.assertEqual(detect_template("Phi-4-mini-instruct-cuda-gpu"), "phi4")
        self.assertEqual(detect_template("qwen2.5-coder-7b-onnx"), "chatml")
        self.assertEqual(detect_template("llama-3.2-3b-instruct-cuda-gpu"), "llama3")
        self.assertEqual(detect_template("deepseek-r1-distill-qwen-7b-onnx"), "deepseek")
        self.assertEqual(detect_template("model", model_type="phi3"), "phi4")
        self.assertEqual(detect_template("something-else"), "chatml")

    def test_none_template_defaults_to_phi4(self):
        self.assertEqual(format_prompt(MSGS, None), format_prompt(MSGS, "phi4"))

    def test_multi_turn_chatml(self):
        out = format_prompt(MSGS, "chatml")
        self.assertTrue(out.startswith("<|im_start|>system\nBe brief.<|im_end|>\n"))
        self.assertIn("<|im_start|>assistant\nHello<|im_end|>", out)
        self.assertTrue(out.endswith("<|im_start|>assistant\n"))

    def test_llama3(self):
        out = format_prompt(MSGS, "llama3")
        self.assertIn("<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>", out)
        self.assertTrue(out.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))

    def test_multiple_system_messages_are_joined(self):
        out = format_prompt([{"role": "system", "content": "A"}, {"role": "system", "content": "B"},
                             {"role": "user", "content": "x"}], "chatml")
        self.assertIn("system\nA\n\nB<|im_end|>", out)


if __name__ == "__main__":
    unittest.main()
