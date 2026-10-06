"""OpenRouter chat completions for the Gemini nodes, with a fallback model.

Gemini answers some photos with PROHIBITED_CONTENT no matter how often you
ask again, and ten retries of the same refusal just stop the queued run. Going
through OpenRouter lets a refusal switch to a different model (Grok refuses far
less) after a couple of tries instead.

Only needs `requests` and Pillow, so it does not import torch or the Gemini SDK.
"""
import base64
import io
import os
import re
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_FALLBACK_MODEL = "x-ai/grok-4.3"

# Bad key or no credits: every model and every retry fails the same way.
_FATAL_STATUS = {401, 402}

# A short answer that only says no. Long answers that happen to contain
# "I can't" mid-prompt are real prompts and are left alone.
_REFUSAL = re.compile(
    r"^\W*(i'?m sorry|sorry|i am sorry|i can(?:'|no)?t|i cannot|i won'?t|i am unable|i'?m unable"
    r"|unable to (?:help|assist|comply)|as an ai)",
    re.IGNORECASE,
)
_REFUSAL_MAX_CHARS = 300

# Optional inputs for the OpenRouter route. Appended after RETRY_INPUTS so
# widget order in existing saved workflows is unchanged.
OPENROUTER_INPUTS = {
    "provider": (["gemini", "openrouter"], {
        "default": "gemini",
        "tooltip": (
            "gemini: call Gemini directly with api_key (the old behaviour). "
            "openrouter: call openrouter_model through OpenRouter with openrouter_api_key."
        ),
    }),
    "openrouter_api_key": ("STRING", {
        "default": "",
        "tooltip": (
            "OpenRouter key (or set OPENROUTER_API_KEY). With provider=gemini it is "
            "only used for the fallback model after Gemini refuses."
        ),
    }),
    "openrouter_model": ("STRING", {
        "default": DEFAULT_MODEL,
        "tooltip": "The model to ask first when provider=openrouter.",
    }),
    "fallback_model": ("STRING", {
        "default": DEFAULT_FALLBACK_MODEL,
        "tooltip": (
            "OpenRouter model to switch to once the first model refused or failed "
            "refusals_before_fallback times. Empty: no fallback."
        ),
    }),
    "refusals_before_fallback": ("INT", {
        "default": 2, "min": 1, "max": 20, "step": 1,
        "tooltip": (
            "Tries with the first model before switching to fallback_model. "
            "The fallback then gets max_retries tries."
        ),
    }),
}


class OpenRouterError(RuntimeError):
    """A failed OpenRouter call. `fatal` means no retry or other model can fix it."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


def resolve_key(api_key: str | None) -> str:
    return (api_key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "")


def image_data_url(pil_image, quality: int = 92) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def is_refusal(text: str) -> bool:
    text = (text or "").strip()
    return not text or (len(text) <= _REFUSAL_MAX_CHARS and bool(_REFUSAL.match(text)))


def build_messages(prompt: str, pil_images: list, system_instruction: str | None = None) -> list:
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_data_url(img)}} for img in pil_images]
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": content})
    return messages


def chat(
    api_key: str,
    model: str,
    messages: list,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    json_mode: bool = False,
    proxy: str | None = None,
    timeout: float = 180,
) -> str:
    """One chat completion. Returns the answer text; raises OpenRouterError."""
    body = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
    if seed is not None:
        body["seed"] = int(seed)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    proxies = {"http": proxy, "https": proxy} if proxy else None

    try:
        r = requests.post(
            API_URL,
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "batch-utility-i9"},
            timeout=timeout,
            proxies=proxies,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(f"{model}: request failed: {exc}") from exc

    if r.status_code >= 400:
        raise OpenRouterError(
            f"{model}: HTTP {r.status_code}: {r.text[:300]}",
            fatal=r.status_code in _FATAL_STATUS,
        )
    try:
        data = r.json()
    except ValueError as exc:
        raise OpenRouterError(f"{model}: not JSON: {r.text[:300]}") from exc

    # OpenRouter reports provider errors (moderation, overload) inside a 200.
    if data.get("error"):
        raise OpenRouterError(f"{model}: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError(f"{model}: no choices in the answer")
    choice = choices[0]
    if choice.get("error"):
        raise OpenRouterError(f"{model}: {choice['error']}")

    text = ((choice.get("message") or {}).get("content") or "").strip()
    if is_refusal(text):
        reason = choice.get("native_finish_reason") or choice.get("finish_reason")
        raise OpenRouterError(f"{model}: refused ({reason}): {text[:120]!r}")
    return text


def chat_with_fallback(
    api_key: str,
    models: list[tuple[str, int]],
    messages: list,
    *,
    seed: int | None,
    logger,
    label: str,
    sleep=time.sleep,
    **chat_kwargs,
) -> str:
    """
    Ask each (model, tries) in turn until one answers, using seed + 1 per try.

    Raises:
        OpenRouterError: every model failed, or the key/credits are bad.
    """
    if not api_key:
        raise OpenRouterError("no OpenRouter key (set openrouter_api_key or OPENROUTER_API_KEY)", fatal=True)
    models = [(m.strip(), max(1, int(n))) for m, n in models if m and m.strip()]
    last_error = None
    attempt = 0

    for index, (model, tries) in enumerate(models):
        for try_no in range(1, tries + 1):
            attempt_seed = None if seed is None else (seed + attempt) % 2**31
            attempt += 1
            try:
                text = chat(api_key, model, messages, seed=attempt_seed, **chat_kwargs)
            except OpenRouterError as exc:
                if exc.fatal:
                    raise
                last_error = exc
                logger.warning(f"{label}: {model} try {try_no}/{tries} failed: {exc}")
                if try_no < tries:
                    sleep(min(2 ** (try_no - 1), 30))
                continue
            if index or try_no > 1:
                logger.info(f"{label}: answered by {model} on try {try_no}")
            return text
        if index + 1 < len(models):
            logger.warning(f"{label}: switching from {model} to {models[index + 1][0]}")

    raise OpenRouterError(f"{label}: no usable answer from {[m for m, _ in models]}: {last_error}")
