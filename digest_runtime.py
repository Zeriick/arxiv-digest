import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

DEFAULT_LLM_BASE_URL = "https://llm-n24dtariayaaxxte.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_LLM_MODEL = "qwen3.7-flash"
DEFAULT_LOG_DIR = "logs"

LOGGER = logging.getLogger("arxiv_digest")
RUN_DIR = None
CLIENT = None


def slugify(value):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "item"


def setup_logging():
    global RUN_DIR

    log_dir = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = log_dir / run_id
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    LOGGER.handlers.clear()
    LOGGER.setLevel(log_level)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(RUN_DIR / "run.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Logging initialized | run_dir=%s", RUN_DIR)


def get_run_dir():
    return RUN_DIR


def write_text_artifact(name, content):
    if RUN_DIR is None:
        return None

    path = RUN_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_json_artifact(name, payload):
    if RUN_DIR is None:
        return None

    path = RUN_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def mask_value(value):
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def get_llm_enable_thinking():
    value = os.getenv("LLM_ENABLE_THINKING", "true").strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise RuntimeError("LLM_ENABLE_THINKING must be a boolean")
    return value in {"1", "true", "yes", "on"}


def get_llm_base_url():
    return os.getenv("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL


def get_client():
    global CLIENT

    if CLIENT is None:
        CLIENT = OpenAI(
            api_key=os.getenv(LLM_API_KEY_ENV),
            base_url=get_llm_base_url(),
        )

    return CLIENT


def create_json_completion(messages, config, temperature=0.2):
    """Assemble streamed answer text, keeping reasoning out of the JSON parser."""
    request = {
        "model": config["llm_model"],
        "messages": messages,
        "temperature": temperature,
        "timeout": config["llm_timeout_seconds"],
        "extra_body": {"enable_thinking": config["llm_enable_thinking"]},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Thinking mode may reject JSON Mode; prompts still require JSON in either mode.
    if not config["llm_enable_thinking"]:
        request["response_format"] = {"type": "json_object"}

    stream = get_client().chat.completions.create(**request)
    parts = []
    usage = None
    response_id = None
    finish_reason = None
    try:
        for chunk in stream:
            response_id = getattr(chunk, "id", None) or response_id
            # The final usage packet has no choices, so read usage before skipping it.
            usage = getattr(chunk, "usage", None) or usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            content = getattr(choice.delta, "content", None)
            if content:
                parts.append(content)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()

    if finish_reason != "stop":
        raise RuntimeError(f"LLM stream did not finish normally: {finish_reason!r}")
    content = "".join(parts)
    if not content.strip():
        raise RuntimeError("LLM stream returned no answer content")

    if usage is not None:
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if hit is None:
            hit = cached
        prompt = getattr(usage, "prompt_tokens", None)
        miss = getattr(usage, "prompt_cache_miss_tokens", None)
        if miss is None and hit is not None and prompt is not None:
            miss = prompt - hit
        usage = SimpleNamespace(
            total_tokens=getattr(usage, "total_tokens", "n/a"),
            prompt_tokens=prompt if prompt is not None else "n/a",
            completion_tokens=getattr(usage, "completion_tokens", "n/a"),
            prompt_cache_hit_tokens=hit if hit is not None else "n/a",
            prompt_cache_miss_tokens=miss if miss is not None else "n/a",
            reasoning_tokens=getattr(
                getattr(usage, "completion_tokens_details", None), "reasoning_tokens", "n/a"
            ),
        )
        LOGGER.info(
            "LLM stream details | model=%s reasoning_tokens=%s response_id=%s",
            config["llm_model"], usage.reasoning_tokens, response_id,
        )
    return SimpleNamespace(
        id=response_id, usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )
