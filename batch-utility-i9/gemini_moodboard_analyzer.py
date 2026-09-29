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
import random

import google.generativeai as genai

from .utils import RETRY_INPUTS, GeminiGenerationError, generate_with_retry, images_to_pillow

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
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "images":          ("IMAGE",),
                "model":           (_MODEL_LIST,),
                "api_key":         ("STRING", {"default": ""}),
                "seed":            ("INT", {
                    "default": seed, "min": 0, "max": 2**31, "step": 1,
                }),
            },
            "optional": {
                "safety_settings": (
                    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],
                    {"default": "BLOCK_NONE"},
                ),
                "temperature":     ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "proxy":           ("STRING", {"default": ""}),
                **RETRY_INPUTS,
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
        api_key: str = "",
        seed: int = 0,
        safety_settings: str = "BLOCK_NONE",
        temperature: float = 0.6,
        proxy: str = "",
        max_retries: int = 5,
        on_failure: str = "stop",
    ) -> tuple:
        # Defensive unwrap for optional scalar inputs
        for name, val, default in [
            ("safety_settings", safety_settings, "BLOCK_NONE"),
            ("proxy",           proxy,           ""),
        ]:
            if isinstance(val, list):
                locals()[name] = val[0] if val else default
        if isinstance(temperature, list):
            temperature = temperature[0] if temperature else 0.6
        if isinstance(api_key, list):
            api_key = api_key[0] if api_key else ""
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
        if isinstance(max_retries, list):
            max_retries = max_retries[0] if max_retries else 5
        if isinstance(on_failure, list):
            on_failure = on_failure[0] if on_failure else "stop"

        api_key = (api_key or "").strip()
        proxy   = (proxy   or "").strip() or None

        if api_key:
            genai.configure(api_key=api_key, transport="rest")
        else:
            genai.configure(transport="rest")

        model_instance = genai.GenerativeModel(
            model,
            safety_settings=safety_settings,
        )

        cfg_kwargs = dict(
            response_mime_type="text/plain",
            temperature=temperature,
            max_output_tokens=2048,
        )

        pil_images = images_to_pillow(images)
        batch_size  = len(pil_images)
        logger.info(f"Analyzing moodboard: {batch_size} image(s) via {model} (seed={seed})")

        try:
            analysis = generate_with_retry(
                model_instance,
                [_ANALYSIS_PROMPT] + pil_images,
                cfg_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logger,
                label="Moodboard analysis",
            )
        except GeminiGenerationError as exc:
            if on_failure == "stop":
                raise
            logger.error(f"{exc} – returning placeholder analysis")
            return (f"[Moodboard analysis failed: {exc}]",)

        logger.info(f"✓ Moodboard analysis: {len(analysis)} chars")
        return (analysis,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiMoodboardAnalyzer": GeminiMoodboardAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiMoodboardAnalyzer": "Gemini Moodboard Analyzer",
}
