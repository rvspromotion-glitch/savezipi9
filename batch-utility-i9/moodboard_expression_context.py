"""
MoodboardExpressionContext
==========================
Enriches a moodboard analysis with targeted expression direction.

Sits between GeminiMoodboardAnalyzer and GeminiCarouselPoseInventor.
Sends the moodboard images back to Gemini with a focused prompt on
facial expressions, gaze, mouth positions, and hand-to-face gestures,
then appends an EXPRESSION DIRECTION section to the analysis text.

The pose director uses this richer context to calibrate how expressions
should look in each invented pose.

Pipeline position:

  GeminiMoodboardAnalyzer → moodboard_analysis
                                    ↓
                     MoodboardExpressionContext  ← moodboard_images
                                    ↓
                       enriched moodboard_analysis
                                    ↓
                     GeminiCarouselPoseInventor
"""

import logging
import random

import google.generativeai as genai

from .utils import RETRY_INPUTS, GeminiGenerationError, generate_with_retry, images_to_pillow

logger = logging.getLogger("MoodboardExpressionContext")

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

_EXPRESSION_PROMPT = """\
You are a facial expression and body language analyst briefing a photo creative director.

Study all the provided images carefully and produce a dense, specific expression brief in EXACTLY this format — use the section headers verbatim:

EXPRESSION DIRECTION:
- DOMINANT ENERGY: [One tight phrase — e.g. "soft and understated", "high-wattage joyful", "cool unbothered", "fierce and intense"]
- MOUTH POSITIONS: [Which mouth states appear and how often — e.g. open laugh, closed neutral, pout, smile, tongue out, parted lips, lip bite. List what you actually see.]
- EYE & GAZE: [Direction (direct camera, downward, sideways, eyes closed), intensity (soft, fierce, vacant, sharp), squinting or half-lidding. Be specific.]
- HAND-TO-FACE: [Whether hands appear near the face and what gestures — peace signs, cupping cheek, covering mouth, finger at lips, hands in hair. Write "none visible" if absent.]
- MIRROR SELFIE: [Yes or No. If yes, describe the expression dynamic: is it posed, candid, cheeky?]
- POSE DIRECTOR NOTES: [2–3 concrete directives the pose director must follow when assigning expressions to invented poses. Example: "keep at least one open-mouth laugh", "avoid both eyes fully closed", "one pose should have direct neutral gaze with closed lips".]

Only describe what is genuinely visible in the images. Do not guess or invent.
"""


class MoodboardExpressionContext:
    """
    Analyzes moodboard images specifically for facial expression patterns
    and appends an EXPRESSION DIRECTION section to the existing moodboard_analysis.

    The enriched string output is a drop-in replacement for moodboard_analysis —
    plug it directly into GeminiCarouselPoseInventor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "moodboard_images": ("IMAGE",),
                "moodboard_analysis": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Connect the output of GeminiMoodboardAnalyzer here.",
                }),
                "model":   (_MODEL_LIST,),
                "api_key": ("STRING", {"default": ""}),
                "seed":    ("INT", {
                    "default": seed, "min": 0, "max": 2**31, "step": 1,
                }),
            },
            "optional": {
                "safety_settings": (
                    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],
                    {"default": "BLOCK_NONE"},
                ),
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower = more faithful to what the images show.",
                }),
                "proxy": ("STRING", {"default": ""}),
                **RETRY_INPUTS,
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("moodboard_analysis",)
    FUNCTION      = "enrich"
    CATEGORY      = "Gemini/Creative"

    def enrich(
        self,
        moodboard_images,
        moodboard_analysis: str,
        model: str,
        api_key: str = "",
        seed: int = 0,
        safety_settings: str = "BLOCK_NONE",
        temperature: float = 0.4,
        proxy: str = "",
        max_retries: int = 5,
        on_failure: str = "stop",
    ) -> tuple:
        if isinstance(api_key, list):
            api_key = api_key[0] if api_key else ""
        if isinstance(moodboard_analysis, list):
            moodboard_analysis = moodboard_analysis[0] if moodboard_analysis else ""
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
        if isinstance(safety_settings, list):
            safety_settings = safety_settings[0] if safety_settings else "BLOCK_NONE"
        if isinstance(temperature, list):
            temperature = temperature[0] if temperature else 0.4
        if isinstance(proxy, list):
            proxy = proxy[0] if proxy else ""
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
            max_output_tokens=1024,
        )

        pil_images = images_to_pillow(moodboard_images)
        logger.info(
            f"Enriching moodboard with expression context: "
            f"{len(pil_images)} image(s) via {model} (seed={seed})"
        )

        try:
            expression_section = generate_with_retry(
                model_instance,
                [_EXPRESSION_PROMPT] + pil_images,
                cfg_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logger,
                label="Expression context",
            )
        except GeminiGenerationError as exc:
            if on_failure == "stop":
                raise
            logger.error(f"{exc} – passing the original analysis through unchanged")
            return (moodboard_analysis,)

        enriched = moodboard_analysis.strip() + "\n\n" + expression_section
        logger.info(f"✓ Expression direction appended: {len(expression_section)} chars")
        return (enriched,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MoodboardExpressionContext": MoodboardExpressionContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoodboardExpressionContext": "Moodboard Expression Context",
}
