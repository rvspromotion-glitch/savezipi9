"""
GeminiMoodboardAnalyzer
=======================
Sends an entire batch of images to Gemini in a single request and returns
a structured creative analysis — aesthetic, photography style, movement,
expression, styling, and a generation-ready style string.

Intended as Stage 1 of a two-stage creative pipeline:

  Batch Image Loader → GeminiMoodboardAnalyzer → moodboard_analysis STRING
                                                ↓
                                       GeminiCarouselPoseInventor
"""

import logging
import os
import time

import google.generativeai as genai

from .utils import images_to_pillow, temporary_env_var

logger = logging.getLogger("GeminiMoodboardAnalyzer")

_MODEL_LIST = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
]

_ANALYSIS_PROMPT = """\
You are a professional creative director, photographer, and fashion stylist.
Analyze all of the images provided (they form a single moodboard) and produce
a structured creative brief in EXACTLY this format — use the section headers
verbatim, keep each section to 2–4 sentences of dense, specific detail:

AESTHETIC & MOOD:
[Overall visual tone, atmosphere, emotional register. Light vs. dark, warm vs. cool, intimate vs. editorial.]

PHOTOGRAPHY STYLE:
[Lens choice, depth of field, lighting setup (natural / strobe / practicals), colour grading, film grain or digital clean.]

MOVEMENT & BODY ENERGY:
[How the subject moves or holds themselves. Energy level (languid, electric, grounded). Intentionality and spontaneity balance.]

EXPRESSION PALETTE:
[Range of facial expressions across the images. Micro-expressions, gaze direction, emotional states conveyed.]

OUTFIT & STYLING DNA:
[Clothing silhouette, texture, colour palette. Accessories, jewellery, shoes if visible. Hair and make-up aesthetic.]

GENERATION-READY STYLE STRING:
[A single dense comma-separated string of style keywords — 30–50 words — that can be appended directly to a Flux/SD prompt to reproduce this visual mood. Cover lighting, colour grade, mood, photography style, and aesthetic. No sentences, only keywords and short phrases.]
"""


class GeminiMoodboardAnalyzer:
    """
    Sends all images in the batch to Gemini at once and returns a structured
    moodboard analysis covering aesthetics, photography, movement, expression,
    styling, and a generation-ready style string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":          ("IMAGE",),
                "model":           (_MODEL_LIST,),
            },
            "optional": {
                "gemini_api_key":  ("STRING", {
                    "default":  "",
                    "multiline": False,
                    "tooltip":  "Falls back to GEMINI_API_KEY / GOOGLE_API_KEY env var if empty.",
                }),
                "safety_settings": (
                    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],
                    {"default": "BLOCK_NONE"},
                ),
                "proxy":           ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("moodboard_analysis",)
    FUNCTION      = "analyze"
    CATEGORY      = "Gemini/Creative"

    # ------------------------------------------------------------------

    def analyze(
        self,
        images,
        model: str,
        gemini_api_key: str = "",
        safety_settings: str = "BLOCK_NONE",
        proxy: str = "",
    ) -> tuple:
        # Defensive unwrap
        if isinstance(gemini_api_key, list):
            gemini_api_key = gemini_api_key[0] if gemini_api_key else ""
        if isinstance(safety_settings, list):
            safety_settings = safety_settings[0] if safety_settings else "BLOCK_NONE"
        if isinstance(proxy, list):
            proxy = proxy[0] if proxy else ""

        gemini_api_key = gemini_api_key.strip()
        proxy          = proxy.strip() or None

        # Resolve API key: widget → GEMINI_API_KEY → GOOGLE_API_KEY
        effective_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if effective_key:
            genai.configure(api_key=effective_key, transport="rest")
        else:
            genai.configure(transport="rest")

        model_instance = genai.GenerativeModel(
            model,
            safety_settings=safety_settings,
        )
        generation_config = genai.GenerationConfig(
            response_mime_type="text/plain",
            temperature=0.6,
            max_output_tokens=2048,
        )

        pil_images = images_to_pillow(images)
        batch_size  = len(pil_images)
        logger.info(f"Analyzing moodboard: {batch_size} image(s) via {model}")

        max_retries = 3
        last_error  = None

        for attempt in range(1, max_retries + 1):
            try:
                # Send ALL images in one request
                content = [_ANALYSIS_PROMPT] + pil_images
                with temporary_env_var("HTTP_PROXY", proxy), \
                     temporary_env_var("HTTPS_PROXY", proxy):
                    response = model_instance.generate_content(
                        content,
                        generation_config=generation_config,
                    )
                analysis = response.text.strip()
                logger.info(
                    f"✓ Moodboard analysis complete: {len(analysis)} chars "
                    f"(attempt {attempt})"
                )
                return (analysis,)

            except Exception as exc:
                last_error = exc
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(1.5)

        logger.error(f"All {max_retries} attempts failed: {last_error}", exc_info=True)
        return (f"[Moodboard analysis failed after {max_retries} attempts: {last_error}]",)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiMoodboardAnalyzer": GeminiMoodboardAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiMoodboardAnalyzer": "Gemini Moodboard Analyzer",
}
