import json
import os
import shutil
import tempfile
import unittest

from prism.templates import (classify_chat_template, detect_template, flatten_content, format_prompt, read_chat_template,
                             resolve_template)

# Chat templates copied from Microsoft's ONNX Runtime GenAI models (the same files `prism pull` downloads).
PHI35_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}"
    "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}"
    "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}")
PHI4_TEMPLATE = (
    "{% for message in messages %}{% if (message['role'] == 'system') %}"
    "{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}"
    "{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|><|im_start|>assistant<|im_sep|>'}}"
    "{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}")
PHI4_MINI_TEMPLATE = (
    "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message %}"
    "{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}"
    "{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}")
CHATML_TEMPLATE = "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}{% endfor %}"
LLAMA3_TEMPLATE = "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
DEEPSEEK_TEMPLATE = "{{'<｜User｜>' + message['content']}}{{'<｜Assistant｜>'}}"
GEMMA_TEMPLATE = "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}"

MSGS = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "Bye"},
]


class TestTemplates(unittest.TestCase):
    def test_detect_template(self):
        self.assertEqual(detect_template("Phi-3.5-mini-instruct-cuda-gpu"), "phi4")
        self.assertEqual(detect_template("Phi-4-mini-instruct-cuda-gpu"), "phi4_mini")
        self.assertEqual(detect_template("Phi-4-instruct-cuda-gpu"), "phi4_im")
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

    def test_flatten_content(self):
        self.assertEqual(flatten_content("x"), "x")
        self.assertEqual(flatten_content(None), "")
        parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "b"}]
        self.assertEqual(flatten_content(parts), "a\nb")

    def test_content_parts_do_not_leak_their_repr_into_the_prompt(self):
        out = format_prompt([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "chatml")
        self.assertIn("<|im_start|>user\nhi<|im_end|>", out)

    def test_phi4_family_formats(self):
        m = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        self.assertEqual(format_prompt(m, "phi4_im"),
                         "<|im_start|>system<|im_sep|>S<|im_end|><|im_start|>user<|im_sep|>U<|im_end|>"
                         "<|im_start|>assistant<|im_sep|>")
        self.assertEqual(format_prompt(m, "phi4_mini"), "<|system|>S<|end|><|user|>U<|end|><|assistant|>")
        self.assertEqual(format_prompt(m[1:], "phi4_im"),  # no system message: none is invented
                         "<|im_start|>user<|im_sep|>U<|im_end|><|im_start|>assistant<|im_sep|>")


class TestChatTemplateDetection(unittest.TestCase):
    def test_classify_real_templates(self):
        for text, expected in ((PHI35_TEMPLATE, "phi4"), (PHI4_TEMPLATE, "phi4_im"), (PHI4_MINI_TEMPLATE, "phi4_mini"),
                               (CHATML_TEMPLATE, "chatml"), (LLAMA3_TEMPLATE, "llama3"), (DEEPSEEK_TEMPLATE, "deepseek")):
            self.assertEqual(classify_chat_template(text), expected, text[:60])

    def test_unknown_or_empty_template_is_none(self):
        self.assertIsNone(classify_chat_template(""))
        self.assertIsNone(classify_chat_template(None))
        self.assertIsNone(classify_chat_template(GEMMA_TEMPLATE))


class TestReadChatTemplate(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, name, content):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))

    def test_no_files_is_empty(self):
        self.assertEqual(read_chat_template(self.dir), "")
        self.assertEqual(read_chat_template(os.path.join(self.dir, "missing")), "")

    def test_tokenizer_config_string(self):
        self.write("tokenizer_config.json", {"chat_template": CHATML_TEMPLATE})
        self.assertEqual(read_chat_template(self.dir), CHATML_TEMPLATE)

    def test_tokenizer_config_list_uses_the_default_template(self):
        self.write("tokenizer_config.json", {"chat_template": [
            {"name": "rag", "template": LLAMA3_TEMPLATE}, {"name": "default", "template": CHATML_TEMPLATE}]})
        self.assertEqual(read_chat_template(self.dir), CHATML_TEMPLATE)

    def test_separate_files_win_over_tokenizer_config(self):  # Qwen3 ships chat_template.jinja and no key in the config
        self.write("tokenizer_config.json", {"chat_template": LLAMA3_TEMPLATE})
        self.write("chat_template.json", {"chat_template": DEEPSEEK_TEMPLATE})
        self.assertEqual(read_chat_template(self.dir), DEEPSEEK_TEMPLATE)
        self.write("chat_template.jinja", CHATML_TEMPLATE)
        self.assertEqual(read_chat_template(self.dir), CHATML_TEMPLATE)

    def test_malformed_files_are_ignored(self):
        self.write("tokenizer_config.json", "{not json")
        self.assertEqual(read_chat_template(self.dir), "")
        self.write("tokenizer_config.json", [1, 2])
        self.assertEqual(read_chat_template(self.dir), "")
        self.write("tokenizer_config.json", {"chat_template": 5})
        self.assertEqual(read_chat_template(self.dir), "")

    def test_an_edited_template_is_read_again(self):
        self.write("chat_template.jinja", CHATML_TEMPLATE)
        self.assertEqual(read_chat_template(self.dir), CHATML_TEMPLATE)
        self.write("chat_template.jinja", LLAMA3_TEMPLATE)
        self.assertEqual(read_chat_template(self.dir), LLAMA3_TEMPLATE)

    def test_resolve_prefers_the_template_over_the_name(self):
        # Named like a Phi model, but the shipped template is Phi-4's <|im_sep|> format
        self.write("tokenizer_config.json", {"chat_template": PHI4_TEMPLATE})
        self.assertEqual(resolve_template("my-phi-finetune", "phi3", self.dir), ("phi4_im", "chat_template"))

    def test_resolve_falls_back_to_the_name(self):
        self.assertEqual(resolve_template("qwen2.5-coder", "qwen2", self.dir), ("chatml", "name"))
        self.write("tokenizer_config.json", {"chat_template": GEMMA_TEMPLATE})  # a template Prism has no format for
        self.assertEqual(resolve_template("llama-3.2-3b", "", self.dir), ("llama3", "name"))


if __name__ == "__main__":
    unittest.main()
