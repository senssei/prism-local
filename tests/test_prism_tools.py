import json
import unittest

from prism.tools import arguments_as_objects, parse_tool_calls, to_openai_tool_calls

WEATHER = {"name": "get_weather", "arguments": {"city": "Paris"}}


class TestParseToolCalls(unittest.TestCase):
    def test_hermes_qwen_format(self):
        text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        self.assertEqual(parse_tool_calls(text), ("Let me check.", [WEATHER]))

    def test_several_hermes_calls(self):
        text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>\n<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
        content, calls = parse_tool_calls(text)
        self.assertEqual((content, [c["name"] for c in calls], calls[1]["arguments"]), ("", ["a", "b"], {"x": 1}))

    def test_hermes_call_cut_off_before_its_closing_tag_still_parses(self):  # generation hit max_tokens right after the JSON
        self.assertEqual(parse_tool_calls('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'), ("", [WEATHER]))

    def test_phi4_mini_format(self):
        text = '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]<|/tool_call|>'
        self.assertEqual(parse_tool_calls(text), ("", [WEATHER]))

    def test_mistral_format(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "abc123def"}]'
        self.assertEqual(parse_tool_calls(text), ("", [WEATHER]))

    def test_llama31_python_tag_and_bare_json_with_parameters(self):
        self.assertEqual(parse_tool_calls('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}'), ("", [WEATHER]))
        self.assertEqual(parse_tool_calls('{"name": "get_weather", "parameters": {"city": "Paris"}}'), ("", [WEATHER]))

    def test_openai_spelling_and_string_arguments(self):
        text = '<tool_call>{"function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}}</tool_call>'
        self.assertEqual(parse_tool_calls(text), ("", [WEATHER]))

    def test_ordinary_answers_are_left_alone(self):
        for text in ("It is sunny in Paris.", "", "Use {\"json\": true} like this.", '{"name": 5}',
                     '<tool_call>not json</tool_call>', '<tool_call>{"name": "f", "arguments": [1]}</tool_call>',
                     '[TOOL_CALLS] nonsense', 'The tag <|tool_call|> starts a call.'):
            self.assertEqual(parse_tool_calls(text), (text, []), text)

    def test_bare_json_followed_by_text_is_not_a_call(self):
        text = '{"name": "get_weather", "parameters": {}} and then some prose'
        self.assertEqual(parse_tool_calls(text), (text, []))


class TestOpenAiShape(unittest.TestCase):
    def test_arguments_are_a_json_string_with_an_id(self):
        out = to_openai_tool_calls([WEATHER, {"name": "b"}])
        self.assertTrue(out[0]["id"].startswith("call_") and out[0]["id"] != out[1]["id"])
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(json.loads(out[0]["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(out[1]["function"]["arguments"], "{}")
        self.assertNotIn("index", out[0])
        self.assertEqual([c["index"] for c in to_openai_tool_calls([WEATHER, WEATHER], with_index=True)], [0, 1])

    def test_message_arguments_become_objects_for_the_template(self):
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "not json"}},
                        {"id": "c3", "type": "function", "function": {"name": "h", "arguments": ""}}]}]
        out = arguments_as_objects(messages)
        args = [c["function"]["arguments"] for c in out[1]["tool_calls"]]
        self.assertEqual(args, [{"a": 1}, "not json", {}])
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], '{"a": 1}')  # the input is untouched
        self.assertIs(out[0], messages[0])


if __name__ == "__main__":
    unittest.main()
