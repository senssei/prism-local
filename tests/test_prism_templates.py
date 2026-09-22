import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from prism.templates import (TemplateToolRenderError, classify_chat_template, detect_template, flatten_content, format_prompt,
                             jinja_available, read_chat_template, read_tokenizer_tokens, render_chat_template, render_prompt,
                             resolve_template, template_mode)

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
# Verbatim from microsoft/mistral-7b-instruct-v0.2-ONNX and unsloth/gemma-2-2b-it (tokenizer_config.json).
MISTRAL_V02_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
GEMMA2_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
# Mistral v0.3 (and later) and Gemma 3, cut down to the parts that identify them.
MISTRAL_V03_TEMPLATE = "{{ bos_token }}{% for m in messages %}{{ '[INST] ' + m['content'] + '[/INST]' }}{% endfor %}"
LLAMA2_TEMPLATE = "{% for m in messages %}{{ '[INST] <<SYS>>\n' + m['content'] + '\n<</SYS>>\n\n [/INST]' }}{% endfor %}"
GEMMA_TEMPLATE = "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}"
UNKNOWN_TEMPLATE = "{% for m in messages %}{{ '### ' + m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"

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

    def test_detect_gemma_and_mistral_by_name(self):
        self.assertEqual(detect_template("gemma-2-2b-it-onnx"), "gemma")
        self.assertEqual(detect_template("mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32"), "mistral_v02")
        self.assertEqual(detect_template("Mistral-7B-Instruct-v0.3-onnx"), "mistral")
        self.assertEqual(detect_template("model", model_type="mistral"), "mistral")

    def test_mistral_formats(self):  # same prompts as the real Jinja templates render (BOS is added by the tokenizer)
        m = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}]
        self.assertEqual(format_prompt(m, "mistral_v02"), "[INST] Hi [/INST]Hello</s>[INST] Bye [/INST]")
        self.assertEqual(format_prompt(m, "mistral"), "[INST] Hi[/INST] Hello</s>[INST] Bye[/INST]")
        # no system role: the system prompt leads the last user message
        self.assertEqual(format_prompt([{"role": "system", "content": "S"}] + m, "mistral"),
                         "[INST] Hi[/INST] Hello</s>[INST] S\n\nBye[/INST]")

    def test_gemma_format(self):
        m = [{"role": "system", "content": "S"}, {"role": "user", "content": " Hi "},
             {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}]
        self.assertEqual(format_prompt(m, "gemma"),
                         "<start_of_turn>user\nS\n\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n"
                         "<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n")

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
                               (CHATML_TEMPLATE, "chatml"), (LLAMA3_TEMPLATE, "llama3"), (DEEPSEEK_TEMPLATE, "deepseek"),
                               (GEMMA2_TEMPLATE, "gemma"), (MISTRAL_V02_TEMPLATE, "mistral_v02"),
                               (MISTRAL_V03_TEMPLATE, "mistral")):
            self.assertEqual(classify_chat_template(text), expected, text[:60])

    def test_unknown_or_empty_template_is_none(self):
        self.assertIsNone(classify_chat_template(""))
        self.assertIsNone(classify_chat_template(None))
        self.assertIsNone(classify_chat_template(UNKNOWN_TEMPLATE))
        self.assertIsNone(classify_chat_template(LLAMA2_TEMPLATE))  # [INST] but Llama 2's format, not Mistral's


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
        self.write("tokenizer_config.json", {"chat_template": UNKNOWN_TEMPLATE})  # a template Prism has no format for
        self.assertEqual(resolve_template("llama-3.2-3b", "", self.dir), ("llama3", "name"))


CONVERSATION = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}]


class ModelFolder(unittest.TestCase):
    """A model folder holding a chat template and a tokenizer config; `resolved` is what the catalog would return for it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def folder(self, template, template_name, **tokenizer_config):
        with open(os.path.join(self.dir, "tokenizer_config.json"), "w") as f:
            json.dump({"chat_template": template, **tokenizer_config}, f)
        return {"id": "m", "path": self.dir, "template": template_name}


@unittest.skipUnless(jinja_available(), "needs the optional jinja2")
class TestJinjaRendering(ModelFolder):
    def test_rendered_prompt_equals_the_builtin_format_for_real_templates(self):
        for text, name, tokens in ((PHI35_TEMPLATE, "phi4", {"eos_token": "<|endoftext|>"}),
                                   (PHI4_TEMPLATE, "phi4_im", {}),
                                   (PHI4_MINI_TEMPLATE, "phi4_mini", {"eos_token": "<|endoftext|>"}),
                                   (MISTRAL_V02_TEMPLATE, "mistral_v02", {"bos_token": "<s>", "eos_token": "</s>", "add_bos_token": True}),
                                   (GEMMA2_TEMPLATE, "gemma", {"bos_token": "<bos>", "add_bos_token": True})):
            resolved = self.folder(text, name, **tokens)
            self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, name), name)

    def test_leading_bos_is_dropped_only_when_the_tokenizer_adds_it(self):
        text = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
        resolved = self.folder(text, "chatml", bos_token="<s>", add_bos_token=True)
        self.assertEqual(render_prompt(resolved, CONVERSATION), "HiHelloBye")
        resolved = self.folder(text, "chatml", bos_token="<s>", add_bos_token=False)
        self.assertEqual(render_prompt(resolved, CONVERSATION), "<s>HiHelloBye")

    def test_bos_token_may_be_an_object(self):
        self.folder("x", "chatml", bos_token={"content": "<|begin|>"}, eos_token={"content": "<|end|>"}, add_bos_token=True)
        self.assertEqual(read_tokenizer_tokens(self.dir), ("<|begin|>", "<|end|>", True))

    def test_a_template_that_refuses_the_conversation_falls_back_to_the_builtin_format(self):
        system = [{"role": "system", "content": "S"}] + CONVERSATION[:1]
        resolved = self.folder(GEMMA2_TEMPLATE, "gemma", bos_token="<bos>", add_bos_token=True)  # Gemma 2 raises on a system role
        with self.assertLogs("prism.templates", "WARNING"):
            out = render_prompt(resolved, system)
        self.assertEqual(out, format_prompt(system, "gemma"))

    def test_the_sandbox_blocks_attribute_tricks(self):
        evil = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        with self.assertRaises(Exception):
            render_chat_template(evil, CONVERSATION)
        resolved = self.folder(evil, "chatml")
        with self.assertLogs("prism.templates", "WARNING"):
            self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, "chatml"))

    def test_tools_reach_the_template_and_tojson_is_not_html_escaped(self):
        text = "{% for t in tools %}{{ t | tojson }}{% endfor %}"
        out = render_chat_template(text, [], tools=[{"name": "a<b", "é": 1}])
        self.assertEqual(out, '{"name": "a<b", "é": 1}')

    def test_content_parts_are_flattened_before_rendering(self):
        resolved = self.folder(CHATML_TEMPLATE, "chatml")
        out = render_prompt(resolved, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(out, "<|im_start|>user\nhi<|im_end|>\n")


class TestTemplateModes(ModelFolder):
    def test_builtin_mode_ignores_the_models_template(self):
        resolved = self.folder("{{ 'RENDERED' }}", "chatml")
        with patch.dict(os.environ, {"PRISM_TEMPLATE": "builtin"}):
            self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, "chatml"))

    def test_without_jinja2_the_builtin_format_is_used(self):
        resolved = self.folder("{{ 'RENDERED' }}", "chatml")
        with patch("prism.templates.jinja_available", return_value=False):
            self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, "chatml"))
            with patch.dict(os.environ, {"PRISM_TEMPLATE": "jinja"}), self.assertLogs("prism.templates", "WARNING"):
                self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, "chatml"))

    def test_a_model_without_a_template_uses_the_builtin_format(self):
        self.assertEqual(render_prompt({"id": "m", "path": self.dir, "template": "llama3"}, CONVERSATION),
                         format_prompt(CONVERSATION, "llama3"))
        self.assertEqual(render_prompt({"id": "m", "template": "llama3"}, CONVERSATION),  # no path at all
                         format_prompt(CONVERSATION, "llama3"))

    def test_template_mode_validation(self):
        with patch.dict(os.environ, {"PRISM_TEMPLATE": "nope"}):
            with self.assertRaises(ValueError):
                template_mode()
        with patch.dict(os.environ, {"PRISM_TEMPLATE": " JINJA "}):
            self.assertEqual(template_mode(), "jinja")


# A template that mentions `tools` (so `supports_tools` will not reject it) but errors at render time when a
# tool definition is actually present. The `{% if tools %}` guard makes the failure conditional on tools being
# passed, which is the exact path Prism must NOT silently fall back from. (`raise_exception` is the template-
# level refusal hook the same way HF templates signal "I refuse this conversation".)
BROKEN_WITH_TOOLS_TEMPLATE = (
    "{% if tools %}{{ raise_exception('template does not accept these tool definitions') }}{% endif %}"
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}")
# A template that raises unconditionally — used to prove the no-tools path still falls back to the built-in
# format (I7 carve-out: silent fallback is preserved when the caller did not ask for tools).
ALWAYS_FAILS_TEMPLATE = "{{ raise_exception('boom') }}"
# A template that takes tools normally; used to prove the new exception is not over-caught.
GOOD_TOOLS_TEMPLATE = (
    "{% if tools %}TOOLS={% for t in tools %}{{ t.function.name }}{% endfor %};{% endif %}"
    "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}")
TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


@unittest.skipUnless(jinja_available(), "needs the optional jinja2")
class TestTemplateToolRenderError(ModelFolder):
    def test_render_prompt_without_tools_still_falls_back_on_template_error(self):
        # I7 carve-out: when the caller did not ask for tools, a template that fails unconditionally still
        # falls back to the built-in format (the messages-only path is unchanged).
        resolved = self.folder(ALWAYS_FAILS_TEMPLATE, "chatml")
        with self.assertLogs("prism.templates", "WARNING"):
            self.assertEqual(render_prompt(resolved, CONVERSATION), format_prompt(CONVERSATION, "chatml"))

    def test_render_prompt_with_tools_raises_template_tool_render_error(self):
        resolved = self.folder(BROKEN_WITH_TOOLS_TEMPLATE, "chatml")
        with self.assertRaises(TemplateToolRenderError) as ctx:
            render_prompt(resolved, CONVERSATION, tools=[TOOL])
        self.assertIsNotNone(ctx.exception.__cause__)
        self.assertIn("m", str(ctx.exception))  # model id (resolved["id"] == "m")

    def test_render_prompt_with_tools_returns_prompt_when_template_accepts_tools(self):
        # The new exception must not fire on templates that take tools normally; guard against over-catching.
        resolved = self.folder(GOOD_TOOLS_TEMPLATE, "chatml")
        out = render_prompt(resolved, CONVERSATION, tools=[TOOL])
        self.assertIn("TOOLS=get_weather;", out)


if __name__ == "__main__":
    unittest.main()
