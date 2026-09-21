import os
from zoneinfo import ZoneInfo

from digest_runtime import (
    DEFAULT_LLM_MODEL,
    LOGGER,
    VALID_LLM_REASONING_EFFORTS,
    get_llm_reasoning_effort,
    mask_value,
    write_json_artifact,
)

DEFAULT_ARXIV_PAGE_SIZE = 100
DEFAULT_MAX_SELECTED_PAPERS = 10
DEFAULT_OPENALEX_TIMEOUT_SECONDS = 15
DEFAULT_OPENALEX_MAX_WORKERS = 8
DEFAULT_LLM_ASSESS_MAX_WORKERS = 16
DEFAULT_LLM_SUMMARY_MAX_WORKERS = 8


def bool_env(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def int_env(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw_value!r}") from exc


def get_runtime_config():
    return {
        "dry_run": bool_env("DRY_RUN", False),
        "log_raw_llm": bool_env("LOG_RAW_LLM", False),
        "llm_model": os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL,
        "llm_timeout_seconds": int_env("LLM_TIMEOUT_SECONDS", 90),
        "llm_reasoning_effort": get_llm_reasoning_effort(),
        "llm_assess_max_workers": int_env(
            "LLM_ASSESS_MAX_WORKERS",
            DEFAULT_LLM_ASSESS_MAX_WORKERS,
        ),
        "llm_summary_max_workers": int_env(
            "LLM_SUMMARY_MAX_WORKERS",
            DEFAULT_LLM_SUMMARY_MAX_WORKERS,
        ),
        "max_selected_papers": int_env(
            "MAX_SELECTED_PAPERS",
            DEFAULT_MAX_SELECTED_PAPERS,
        ),
        "arxiv_page_size": int_env(
            "ARXIV_PAGE_SIZE",
            DEFAULT_ARXIV_PAGE_SIZE,
        ),
        "target_days_ago": int_env("TARGET_DAYS_AGO", 1),
        "local_timezone": os.getenv("LOCAL_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai",
        "openalex_enrichment_enabled": bool_env("OPENALEX_ENRICHMENT_ENABLED", True),
        "openalex_timeout_seconds": int_env(
            "OPENALEX_TIMEOUT_SECONDS",
            DEFAULT_OPENALEX_TIMEOUT_SECONDS,
        ),
        "openalex_max_workers": int_env(
            "OPENALEX_MAX_WORKERS",
            DEFAULT_OPENALEX_MAX_WORKERS,
        ),
        "openalex_email": os.getenv("OPENALEX_EMAIL", "").strip(),
    }


def get_smtp_config():
    return {
        "host": os.getenv("EMAIL_SMTP_HOST", "mail.tiaozhan.com"),
        "port": int_env("EMAIL_SMTP_PORT", 465),
        "use_ssl": bool_env("EMAIL_USE_SSL", True),
        "use_starttls": bool_env("EMAIL_USE_STARTTLS", False),
        "user": os.getenv("EMAIL_USER", "").strip(),
        "password": os.getenv("EMAIL_PASS", ""),
        "to": os.getenv("EMAIL_TO", "").strip(),
    }


def validate_runtime_config(config, smtp_config):
    missing = []

    if not os.getenv("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")

    if not config["dry_run"]:
        for name, value in {
            "EMAIL_USER": smtp_config["user"],
            "EMAIL_PASS": smtp_config["password"],
            "EMAIL_TO": smtp_config["to"],
        }.items():
            if not value:
                missing.append(name)

    if config["max_selected_papers"] <= 0:
        raise RuntimeError("MAX_SELECTED_PAPERS must be greater than 0")

    if config["arxiv_page_size"] <= 0:
        raise RuntimeError("ARXIV_PAGE_SIZE must be greater than 0")

    if config["target_days_ago"] <= 0:
        raise RuntimeError("TARGET_DAYS_AGO must be greater than 0")

    if config["llm_timeout_seconds"] <= 0:
        raise RuntimeError("LLM_TIMEOUT_SECONDS must be greater than 0")

    if config["llm_reasoning_effort"] not in VALID_LLM_REASONING_EFFORTS:
        allowed = ", ".join(sorted(VALID_LLM_REASONING_EFFORTS))
        raise RuntimeError(
            f"LLM_REASONING_EFFORT must be one of: {allowed}, "
            f"got {config['llm_reasoning_effort']!r}"
        )

    if config["llm_assess_max_workers"] <= 0:
        raise RuntimeError("LLM_ASSESS_MAX_WORKERS must be greater than 0")

    if config["llm_summary_max_workers"] <= 0:
        raise RuntimeError("LLM_SUMMARY_MAX_WORKERS must be greater than 0")

    if config["openalex_timeout_seconds"] <= 0:
        raise RuntimeError("OPENALEX_TIMEOUT_SECONDS must be greater than 0")

    if config["openalex_max_workers"] <= 0:
        raise RuntimeError("OPENALEX_MAX_WORKERS must be greater than 0")

    try:
        ZoneInfo(config["local_timezone"])
    except Exception as exc:
        raise RuntimeError(f"LOCAL_TIMEZONE is invalid: {config['local_timezone']!r}") from exc

    if smtp_config["use_ssl"] and smtp_config["use_starttls"]:
        raise RuntimeError("EMAIL_USE_SSL and EMAIL_USE_STARTTLS cannot both be true")

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def log_runtime_config(config, smtp_config):
    safe_config = {
        "dry_run": config["dry_run"],
        "log_raw_llm": config["log_raw_llm"],
        "llm_model": config["llm_model"],
        "llm_timeout_seconds": config["llm_timeout_seconds"],
        "llm_reasoning_effort": config["llm_reasoning_effort"],
        "llm_assess_max_workers": config["llm_assess_max_workers"],
        "llm_summary_max_workers": config["llm_summary_max_workers"],
        "max_selected_papers": config["max_selected_papers"],
        "arxiv_page_size": config["arxiv_page_size"],
        "target_days_ago": config["target_days_ago"],
        "local_timezone": config["local_timezone"],
        "openalex_enrichment_enabled": config["openalex_enrichment_enabled"],
        "openalex_timeout_seconds": config["openalex_timeout_seconds"],
        "openalex_max_workers": config["openalex_max_workers"],
        "openalex_email": config["openalex_email"],
        "smtp_host": smtp_config["host"],
        "smtp_port": smtp_config["port"],
        "smtp_use_ssl": smtp_config["use_ssl"],
        "smtp_use_starttls": smtp_config["use_starttls"],
        "email_user": smtp_config["user"],
        "email_to": smtp_config["to"],
        "deepseek_api_key_masked": mask_value(os.getenv("DEEPSEEK_API_KEY", "")),
        "email_pass_masked": mask_value(smtp_config["password"]),
    }
    LOGGER.info(
        "Runtime configuration loaded | dry_run=%s model=%s llm_timeout=%ss reasoning_effort=%s llm_assess_workers=%s llm_summary_workers=%s max_selected=%s arxiv_page_size=%s target_days_ago=%s timezone=%s openalex_enabled=%s openalex_timeout=%ss openalex_workers=%s smtp_host=%s smtp_port=%s smtp_ssl=%s smtp_starttls=%s log_raw_llm=%s",
        safe_config["dry_run"],
        safe_config["llm_model"],
        safe_config["llm_timeout_seconds"],
        safe_config["llm_reasoning_effort"],
        safe_config["llm_assess_max_workers"],
        safe_config["llm_summary_max_workers"],
        safe_config["max_selected_papers"],
        safe_config["arxiv_page_size"],
        safe_config["target_days_ago"],
        safe_config["local_timezone"],
        safe_config["openalex_enrichment_enabled"],
        safe_config["openalex_timeout_seconds"],
        safe_config["openalex_max_workers"],
        safe_config["smtp_host"],
        safe_config["smtp_port"],
        safe_config["smtp_use_ssl"],
        safe_config["smtp_use_starttls"],
        safe_config["log_raw_llm"],
    )
    write_json_artifact("config.json", safe_config)
