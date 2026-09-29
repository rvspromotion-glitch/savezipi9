import os
import time
from contextlib import contextmanager
from typing import Callable, Generator

import google.generativeai as genai
import torch
from PIL import Image

try:
    from google.api_core import exceptions as _api_exceptions

    # Errors that will fail identically no matter how often we retry
    # (bad API key, unknown model, malformed request, no access).
    _FATAL_ERRORS: tuple = (
        _api_exceptions.InvalidArgument,
        _api_exceptions.Unauthenticated,
        _api_exceptions.PermissionDenied,
        _api_exceptions.NotFound,
    )
except ImportError:
    _FATAL_ERRORS = ()


# Optional inputs shared by every Gemini node.  Appended at the END of the
# optional inputs so widget order in existing saved workflows is unchanged.
RETRY_INPUTS = {
    "max_retries": ("INT", {
        "default": 5, "min": 1, "max": 20, "step": 1,
        "tooltip": (
            "How many times to call Gemini before giving up. "
            "Each retry uses seed + 1 and waits a little longer (1s, 2s, 4s, … max 30s)."
        ),
    }),
    "on_failure": (["stop", "fallback"], {
        "default": "stop",
        "tooltip": (
            "stop: raise an error and halt the queued run when Gemini returns nothing "
            "after all retries. fallback: continue with a placeholder text instead."
        ),
    }),
}


class GeminiGenerationError(RuntimeError):
    """Raised when Gemini produced no usable text after all retries."""


def images_to_pillow(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert ComfyUI image tensor(s) to PIL Image(s).
    
    Args:
        images: Tensor of shape [B, H, W, C] where B is batch size
        
    Returns:
        List of PIL Images
    """
    pil_images = []
    
    # Handle batch of images
    for i in range(images.shape[0]):
        img_tensor = images[i]
        
        # Convert from [H, W, C] tensor (0-1 float) to PIL Image
        # ComfyUI uses 0-1 range, convert to 0-255
        img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
        pil_img = Image.fromarray(img_array)
        pil_images.append(pil_img)
    
    return pil_images


@contextmanager
def temporary_env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """
    Temporarily set an environment variable.
    
    Args:
        key: Environment variable name
        value: Value to set (or None to skip)
        
    Yields:
        None
    """
    if value is None:
        yield
        return
    
    old_value = os.environ.get(key)
    
    try:
        os.environ[key] = value
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def make_generation_config(seed: int | None = None, **kwargs) -> genai.GenerationConfig:
    """Build a GenerationConfig, dropping `seed` on SDK versions that don't support it."""
    if seed is not None:
        try:
            return genai.GenerationConfig(**kwargs, seed=seed)
        except TypeError:
            pass
    return genai.GenerationConfig(**kwargs)


def _describe_empty_response(response) -> str:
    """Explain why a response has no text (blocked prompt, finish reason, …)."""
    details = []
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    if block_reason:
        details.append(f"prompt blocked: {getattr(block_reason, 'name', block_reason)}")
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        details.append(f"finish_reason={getattr(finish_reason, 'name', finish_reason)}")
    else:
        details.append("no candidates")
    return f"empty response ({', '.join(details)})"


def response_text(response) -> str:
    """Return the stripped response text, raising ValueError if there is none."""
    try:
        text = response.text
    except ValueError as exc:
        raise ValueError(_describe_empty_response(response)) from exc
    text = (text or "").strip()
    if not text:
        raise ValueError(_describe_empty_response(response))
    return text


def generate_with_retry(
    model_instance,
    contents: list | Callable[[str | None], list],
    config_kwargs: dict,
    *,
    seed: int | None,
    max_retries: int,
    proxy: str | None,
    logger,
    label: str,
    check: Callable[[str, object], str | None] | None = None,
) -> str:
    """
    Call Gemini until it returns usable text, using seed + 1 on every retry.

    Args:
        contents: Request contents, or a callable taking the previous rejected
            text (or None) and returning the contents for the next attempt.
        config_kwargs: GenerationConfig kwargs, without the seed.
        check: Optional callable(text, response) returning a reason string to
            retry. Text that only fails this check is still returned if no
            later attempt does better.

    Raises:
        GeminiGenerationError: No attempt produced any text, or the error is
            one that retrying cannot fix (bad key, unknown model, …).
    """
    max_retries = max(1, int(max_retries))
    last_error = None
    rejected_text = None

    for attempt in range(1, max_retries + 1):
        attempt_seed = None if seed is None else (seed + attempt - 1) % 2**31
        generation_config = make_generation_config(seed=attempt_seed, **config_kwargs)
        request = contents(rejected_text) if callable(contents) else contents

        try:
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                response = model_instance.generate_content(request, generation_config=generation_config)
            text = response_text(response)
        except _FATAL_ERRORS as exc:
            raise GeminiGenerationError(f"{label}: non-retryable Gemini error: {exc}") from exc
        except Exception as exc:
            last_error = exc
            logger.warning(f"{label}: attempt {attempt}/{max_retries} failed (seed={attempt_seed}): {exc}")
        else:
            reason = check(text, response) if check else None
            if reason is None:
                if attempt > 1:
                    logger.info(f"{label}: succeeded on attempt {attempt} (seed={attempt_seed})")
                return text
            rejected_text = text
            last_error = reason
            logger.warning(f"{label}: attempt {attempt}/{max_retries} rejected (seed={attempt_seed}): {reason}")

        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 30))

    if rejected_text is not None:
        logger.warning(f"{label}: keeping best result after {max_retries} attempts ({last_error})")
        return rejected_text

    raise GeminiGenerationError(f"{label}: no usable response after {max_retries} attempts: {last_error}")
