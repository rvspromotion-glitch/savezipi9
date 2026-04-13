"""
GeminiCarouselPoseInventor
==========================
Stage 2 of the creative pipeline.  Takes the moodboard analysis from
GeminiMoodboardAnalyzer and a single reference image, then asks Gemini to
invent N distinct pose sets for a carousel shoot.

Each pose set uses a strict three-line format that maps directly onto a Flux
prompt structure:

    [POSE] ...
    [EXPRESSION] ...
    [CROP & ANGLE] ...

Poses are intentionally simple (one action only) because Flux/SD struggles
with compound poses.  The shooting context (mirror selfie, outdoor, etc.) is
inferred from the reference image and locked so all poses stay plausible.

Outputs
-------
pose_sets_json : STRING  — JSON array, one element per pose set
pose_sets_raw  : STRING  — same content joined by  ---  (for debugging /
                           connecting to a text node directly)

Intended workflow
-----------------
GeminiMoodboardAnalyzer ──► moodboard_analysis ──►┐
                                                    ├──► GeminiCarouselPoseInventor
Ref Image ──────────────────────────────────────►──┘
"""

import json
import logging
import os
import re
import time

import google.generativeai as genai

from .utils import images_to_pillow, temporary_env_var

logger = logging.getLogger("GeminiCarouselPoseInventor")

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

_PROMPT_TEMPLATE = """\
You are a creative director inventing poses for a photo carousel shoot.

=== MOODBOARD ANALYSIS ===
{moodboard_analysis}

=== TASK ===
First, study the reference image carefully to identify the exact shooting context:
- What kind of shot is it? (mirror selfie, outdoor candid, studio, bathroom, car interior, etc.)
- What is the camera angle and distance?
- What environment / props are present?

LOCK the shooting context to match the reference image exactly.  Every pose you invent must be physically plausible in that same environment.

Invent {pose_count} distinct pose sets for a carousel.

=== OUTPUT FORMAT ===
Each pose set must use EXACTLY this three-line format:

[POSE] <single simple action — ONE thing only, never compound>
[EXPRESSION] <one specific facial expression or micro-expression>
[CROP & ANGLE] <framing (close-up / waist-up / full-body) and camera angle>

=== RULES ===
1. [POSE] must describe ONE action only.  Good: "leans back against wall".  Bad: "leans back while reaching up and tilting head".
2. Vary the poses meaningfully — no two poses should feel redundant.
3. All poses must be achievable in the shooting context from the reference image.
4. Keep language concrete and visual — no abstract adjectives.
5. Separate each pose set from the next with a line containing only: ---
6. Do NOT add numbering, headers, or any text outside the pose set blocks.
7. Start your response immediately with the first [POSE] line.
"""


def _parse_pose_sets(raw_text: str, expected_count: int) -> list[str]:
    """
    Parse Gemini's response into a list of pose-set strings.

    Splits on `---` dividers (with or without surrounding whitespace / extra
    dashes) and validates that each block contains all three required tags.
    """
    # Normalise separator variations (---, -----, — etc.)
    normalised = re.sub(r"\n[-—]{2,}\n", "\n---\n", raw_text)
    blocks = [b.strip() for b in normalised.split("---") if b.strip()]

    valid = []
    for block in blocks:
        if "[POSE]" in block and "[EXPRESSION]" in block and "[CROP & ANGLE]" in block:
            valid.append(block)

    if not valid:
        logger.warning("No valid pose-set blocks found; returning raw text as single entry")
        return [raw_text.strip()]

    if len(valid) < expected_count:
        logger.warning(
            f"Expected {expected_count} pose sets but only parsed {len(valid)}; "
            "using what we got"
        )

    return valid


class GeminiCarouselPoseInventor:
    """
    Invents N pose sets for a photo carousel using the moodboard analysis from
    GeminiMoodboardAnalyzer and a reference image that locks the shooting context.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image":           ("IMAGE",),
                "moodboard_analysis":  ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Paste the output of GeminiMoodboardAnalyzer here.",
                }),
                "pose_count": ("INT", {
                    "default": 6,
                    "min":     1,
                    "max":     15,
                    "step":    1,
                    "tooltip": "Number of distinct pose sets to invent.",
                }),
                "model": (_MODEL_LIST,),
            },
            "optional": {
                "gemini_api_key": ("STRING", {
                    "default":   "",
                    "multiline": False,
                    "tooltip":   "Falls back to GEMINI_API_KEY / GOOGLE_API_KEY env var if empty.",
                }),
                "safety_settings": (
                    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],
                    {"default": "BLOCK_NONE"},
                ),
                "proxy": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("pose_sets_json", "pose_sets_raw")
    FUNCTION      = "invent_poses"
    CATEGORY      = "Gemini/Creative"

    # ------------------------------------------------------------------

    def invent_poses(
        self,
        ref_image,
        moodboard_analysis: str,
        pose_count: int = 6,
        model: str = "gemini-2.5-flash",
        gemini_api_key: str = "",
        safety_settings: str = "BLOCK_NONE",
        proxy: str = "",
    ) -> tuple:
        # Defensive unwrap
        if isinstance(gemini_api_key, list):
            gemini_api_key = gemini_api_key[0] if gemini_api_key else ""
        if isinstance(moodboard_analysis, list):
            moodboard_analysis = moodboard_analysis[0] if moodboard_analysis else ""
        if isinstance(pose_count, list):
            pose_count = pose_count[0] if pose_count else 6
        if isinstance(safety_settings, list):
            safety_settings = safety_settings[0] if safety_settings else "BLOCK_NONE"
        if isinstance(proxy, list):
            proxy = proxy[0] if proxy else ""

        gemini_api_key = gemini_api_key.strip()
        proxy          = proxy.strip() or None

        # Resolve API key
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
            temperature=0.9,   # higher → more creative pose variety
            max_output_tokens=4096,
        )

        # Use only the first frame as reference (or all if it's a small batch)
        pil_images = images_to_pillow(ref_image)
        ref_pil    = pil_images[0]

        prompt = _PROMPT_TEMPLATE.format(
            moodboard_analysis=moodboard_analysis.strip(),
            pose_count=pose_count,
        )

        logger.info(
            f"Inventing {pose_count} pose sets via {model} "
            f"(ref image {ref_pil.width}×{ref_pil.height})"
        )

        max_retries = 3
        last_error  = None

        for attempt in range(1, max_retries + 1):
            try:
                content = [prompt, ref_pil]
                with temporary_env_var("HTTP_PROXY", proxy), \
                     temporary_env_var("HTTPS_PROXY", proxy):
                    response = model_instance.generate_content(
                        content,
                        generation_config=generation_config,
                    )
                raw_text = response.text.strip()
                logger.info(
                    f"✓ Raw response: {len(raw_text)} chars (attempt {attempt})"
                )

                pose_sets = _parse_pose_sets(raw_text, pose_count)

                pose_sets_json = json.dumps(pose_sets, ensure_ascii=False, indent=2)
                pose_sets_raw  = "\n---\n".join(pose_sets)

                logger.info(f"✓ Parsed {len(pose_sets)} pose sets")
                for i, ps in enumerate(pose_sets):
                    preview = ps.replace("\n", " | ")[:120]
                    logger.debug(f"  [{i + 1}] {preview}")

                return (pose_sets_json, pose_sets_raw)

            except Exception as exc:
                last_error = exc
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(1.5)

        logger.error(f"All {max_retries} attempts failed: {last_error}", exc_info=True)
        fallback_json = json.dumps(
            [f"[Error] Failed to generate pose sets: {last_error}"],
            ensure_ascii=False,
        )
        return (fallback_json, str(last_error))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiCarouselPoseInventor": GeminiCarouselPoseInventor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiCarouselPoseInventor": "Gemini Carousel Pose Inventor",
}
