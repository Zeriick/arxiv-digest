import os
import json
import unittest
from unittest.mock import patch

from openai.types.chat import ChatCompletionChunk

from digest_config import get_runtime_config, validate_runtime_config
from digest_llm import assess_paper, llm_call, summarize
from digest_runtime import DEFAULT_LLM_BASE_URL, create_json_completion, get_client
from macro_config import get_macro_runtime_config, validate_macro_runtime_config
from macro_llm import call_macro_synthesis_model, repair_macro_json_with_llm


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def chunk(delta=None, finish=None, usage=None):
    return ChatCompletionChunk.model_validate({
        "id": "test-stream", "created": 0, "model": "qwen3.7-flash",
        "object": "chat.completion.chunk",
        "choices": [] if delta is None else [
            {"index": 0, "delta": delta, "finish_reason": finish}
        ],
        "usage": usage,
    })


def answer_stream(usage=None, finish="stop"):
    return FakeStream([
        chunk({"reasoning_content": "Reasoning containing {invalid JSON}"}),
        chunk({"content": '{"ok":'}),
        chunk({"content": "true}"}),
        chunk({}, finish=finish),
        chunk(usage=usage),
    ])


class BailianMigrationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "llm_model": "qwen3.7-flash", "llm_enable_thinking": True,
            "llm_timeout_seconds": 90, "log_raw_llm": False,
        }
        self.messages = [{"role": "user", "content": "Return JSON only."}]

    def test_client_uses_dashscope_key_and_requested_endpoint(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True), \
                patch("digest_runtime.CLIENT", None), patch("digest_runtime.OpenAI") as client:
            get_client()
        client.assert_called_once_with(api_key="test-key", base_url=DEFAULT_LLM_BASE_URL)

    def test_both_configs_default_to_qwen_and_validate_dashscope_credentials(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key", "DRY_RUN": "true"}, clear=True):
            for config, validate in [
                (get_runtime_config(), validate_runtime_config),
                (get_macro_runtime_config(), validate_macro_runtime_config),
            ]:
                self.assertEqual(config["llm_model"], "qwen3.7-flash")
                self.assertTrue(config["llm_enable_thinking"])
                validate(config, {"use_ssl": True, "use_starttls": False})
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                    with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
                        validate(config, {"use_ssl": True, "use_starttls": False})

    def test_summary_model_has_independent_default_and_override(self):
        with patch.dict(os.environ, {"LLM_MODEL": "qwen3.7-flash"}, clear=True):
            self.assertEqual(get_runtime_config()["llm_summary_model"], "qwen3.7-plus")
            with patch.dict(os.environ, {"LLM_SUMMARY_MODEL": "custom-summary-model"}):
                self.assertEqual(get_runtime_config()["llm_summary_model"], "custom-summary-model")

    @patch("digest_runtime.get_client")
    def test_only_selected_paper_summary_uses_plus(self, client):
        assessment = {"relevant": True, "score": 85, "fit_area": "AI-Compiler"}
        summary = {
            "summary": ["Problem", "Method", "Result"], "translation": "中文概述",
            "reason": "Concrete contribution", "affiliation_signal": "No useful signal",
        }
        client.return_value.chat.completions.create.side_effect = [
            FakeStream([chunk({"content": json.dumps(assessment)}, finish="stop")]),
            FakeStream([chunk({"content": json.dumps(summary)}, finish="stop")]),
            answer_stream(),
        ]
        self.assertEqual(assess_paper("Title", "Abstract", [], "paper", self.config), assessment)
        self.assertEqual(summarize("Title", "Abstract", "paper", self.config), summary)
        call_macro_synthesis_model("JSON", self.config)
        models = [call.kwargs["model"] for call in client.return_value.chat.completions.create.call_args_list]
        self.assertEqual(models, ["qwen3.7-flash", "qwen3.7-plus", "qwen3.7-flash"])
        self.assertEqual(self.config["llm_model"], "qwen3.7-flash")

    @patch("digest_runtime.get_client")
    def test_stream_separates_reasoning_and_reads_usage_only_packet(self, client):
        stream = answer_stream({
            "prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 20},
        })
        client.return_value.chat.completions.create.return_value = stream
        response = create_json_completion(self.messages, self.config)
        self.assertEqual(response.choices[0].message.content, '{"ok":true}')
        self.assertEqual(response.usage.prompt_cache_hit_tokens, 60)
        self.assertEqual(response.usage.prompt_cache_miss_tokens, 40)
        self.assertEqual(response.usage.reasoning_tokens, 20)
        self.assertTrue(stream.closed)
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"enable_thinking": True})
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertNotIn("response_format", request)

    @patch("digest_runtime.get_client")
    def test_non_thinking_uses_json_mode_and_unknown_cache_is_not_zero(self, client):
        client.return_value.chat.completions.create.return_value = answer_stream({
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        })
        self.config["llm_enable_thinking"] = False
        response = create_json_completion(self.messages, self.config)
        self.assertEqual(response.usage.prompt_cache_hit_tokens, "n/a")
        self.assertEqual(response.usage.prompt_cache_miss_tokens, "n/a")
        request = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertEqual(request["response_format"], {"type": "json_object"})

    @patch("digest_runtime.get_client")
    def test_truncated_interrupted_and_empty_streams_fail_and_close(self, client):
        for stream in [
            answer_stream(finish="length"),
            FakeStream([chunk({"content": "{"}), OSError("connection lost")]),
            FakeStream([chunk({}, finish="stop")]),
            FakeStream([chunk({"content": "{}"})]),
        ]:
            with self.subTest(stream=stream):
                client.return_value.chat.completions.create.return_value = stream
                with self.assertRaises((RuntimeError, OSError)):
                    create_json_completion(self.messages, self.config)
                self.assertTrue(stream.closed)

    @patch("digest_runtime.get_client")
    def test_digest_macro_and_json_repair_all_use_stream_adapter(self, client):
        client.return_value.chat.completions.create.side_effect = lambda **kwargs: answer_stream()
        self.assertEqual(llm_call("JSON", "assess", "paper", self.config), '{"ok":true}')
        self.assertEqual(call_macro_synthesis_model("JSON", self.config), '{"ok":true}')
        self.assertEqual(repair_macro_json_with_llm("broken", self.config), '{"ok":true}')
        calls = client.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(call.kwargs["model"], "qwen3.7-flash")
            self.assertTrue(call.kwargs["stream"])
            self.assertEqual(call.kwargs["extra_body"], {"enable_thinking": True})
        self.assertEqual(calls[-1].kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
