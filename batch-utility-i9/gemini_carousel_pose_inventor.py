"""
GeminiCarouselPoseInventor
==========================
Stage 2 of the creative pipeline.  Takes the moodboard analysis from
GeminiMoodboardAnalyzer and a single reference image, then asks Gemini to
invent N distinct pose sets for a carousel shoot.

Each pose set uses a strict three-line format:

    [POSE] ...
    [EXPRESSION] ...
    [CROP & ANGLE] ...

Poses are intentionally simple (one action only) because Flux/SD struggles
with compound poses.  The shooting context (mirror selfie, outdoor, etc.) is
inferred from the reference image and locked so all poses stay plausible.

Outputs
-------
pose_sets      : STRING list  — one string per pose set (OUTPUT_IS_LIST)
                               feeds directly into StringConcatBatch / CLIPTextEncodeBatch
pose_sets_raw  : STRING       — all pose sets joined by --- (debug / text preview)
"""

import json
import logging
import random
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

_MIRROR_SELFIE_PROMPT_TEMPLATE = """\
You are a creative director inventing poses for a mirror-selfie photo carousel shoot.

=== MOODBOARD ANALYSIS ===
{moodboard_analysis}

=== SHOOTING CONTEXT — LOCKED ===
This is a MIRROR SELFIE.  The following constraints are absolute and cannot be broken:

1. ONE HAND IS ALWAYS OCCUPIED holding the phone up toward the mirror.
   - Never invent a pose that requires both arms to be free.
   - The phone arm is typically raised to chest / shoulder height.
   - Poses involving the free arm only: resting, touching hair, on hip, on waist, in pocket, touching clothing, etc.

2. THE FACE OR GAZE MUST BE VISIBLE TO THE CAMERA in most poses, but back-to-mirror shots ARE valid — e.g. turned away showing the back/ass while looking over the shoulder at the phone. Only invent back-facing poses when they make physical sense (phone still raised toward mirror).

3. CAMERA ANGLE IS FIXED by the mirror height and phone arm reach.
   - Slight high-angle (phone raised) or straight-on are the only realistic options.
   - Do not invent drone, floor-level, or extreme angles.

4. ENVIRONMENT: match exactly what is visible in the reference image
   (bathroom mirror, bedroom mirror, gym mirror, fitting-room mirror, etc.).
   All props and background elements must be consistent with that space.

Study the reference image to confirm the exact mirror type, room, and phone-arm position, then invent {pose_count} distinct pose sets.

=== OUTPUT FORMAT ===
Each pose set must use EXACTLY this three-line format:

[POSE] <single action with the FREE arm/body only — never requires both hands>
[EXPRESSION] <one specific facial expression or micro-expression>
[CROP & ANGLE] <framing (close-up / waist-up / full-body) — angle must be realistic for a mirror selfie>

=== RULES ===
1. [POSE] describes ONE action only.  Good: "free hand rests on hip".  Bad: "both arms raised above head".
2. Never describe what the phone hand is doing — it is always holding the phone.
3. Vary poses meaningfully — no two should feel redundant.
4. Keep language concrete and visual — no abstract adjectives.
5. Separate each pose set from the next with a line containing only: ---
6. Do NOT add numbering, headers, or any text outside the pose set blocks.
7. Start your response immediately with the first [POSE] line.
"""


def _parse_pose_sets(raw_text: str, expected_count: int) -> list[str]:
    """
    Split Gemini's response into individual pose-set strings.
    Accepts ---, ----, — etc. as separators and validates that each block
    contains all three required tags.
    """
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
            f"Expected {expected_count} pose sets but parsed {len(valid)}"
        )

    return valid


class GeminiCarouselPoseInventor:
    """
    Invents N pose sets for a photo carousel.
    Output pose_sets is a STRING list (one element per pose set) compatible
    with StringConcatBatch and CLIPTextEncodeBatch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "ref_image":          ("IMAGE",),
                "moodboard_analysis": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Connect the output of GeminiMoodboardAnalyzer here.",
                }),
                "pose_count": ("INT", {
                    "default": 6, "min": 1, "max": 15, "step": 1,
                }),
                "model":   (_MODEL_LIST,),
                "api_key": ("STRING", {"default": ""}),
                "seed":    ("INT", {
                    "default": seed, "min": 0, "max": 2**31, "step": 1,
                }),
            },
            "optional": {
                "mirror_selfie_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Use a mirror-selfie-aware prompt that enforces the "
                        "phone-in-hand constraint and mirror-facing body position."
                    ),
                }),
                "safety_settings": (
                    ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],
                    {"default": "BLOCK_NONE"},
                ),
                "temperature": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Higher = more creative pose variety.",
                }),
                "proxy": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("pose_sets", "pose_sets_raw")
    FUNCTION      = "invent_poses"
    CATEGORY      = "Gemini/Creative"
    OUTPUT_IS_LIST = (True, False)   # pose_sets → list; pose_sets_raw → single string

    # ------------------------------------------------------------------

    def invent_poses(
        self,
        ref_image,
        moodboard_analysis: str,
        pose_count: int = 6,
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        seed: int = 0,
        mirror_selfie_mode: bool = False,
        safety_settings: str = "BLOCK_NONE",
        temperature: float = 0.9,
        proxy: str = "",
    ) -> tuple:
        # Defensive unwrap
        if isinstance(api_key, list):
            api_key = api_key[0] if api_key else ""
        if isinstance(moodboard_analysis, list):
            moodboard_analysis = moodboard_analysis[0] if moodboard_analysis else ""
        if isinstance(pose_count, list):
            pose_count = pose_count[0] if pose_count else 6
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
        if isinstance(mirror_selfie_mode, list):
            mirror_selfie_mode = mirror_selfie_mode[0] if mirror_selfie_mode else False
        if isinstance(safety_settings, list):
            safety_settings = safety_settings[0] if safety_settings else "BLOCK_NONE"
        if isinstance(temperature, list):
            temperature = temperature[0] if temperature else 0.9
        if isinstance(proxy, list):
            proxy = proxy[0] if proxy else ""

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
            max_output_tokens=4096,
        )
        try:
            generation_config = genai.GenerationConfig(**cfg_kwargs, seed=seed)
        except TypeError:
            generation_config = genai.GenerationConfig(**cfg_kwargs)

        pil_images = images_to_pillow(ref_image)
        ref_pil    = pil_images[0]

        template = _MIRROR_SELFIE_PROMPT_TEMPLATE if mirror_selfie_mode else _PROMPT_TEMPLATE
        prompt = template.format(
            moodboard_analysis=moodboard_analysis.strip(),
            pose_count=pose_count,
        )

        logger.info(
            f"Inventing {pose_count} pose sets via {model} "
            f"(seed={seed}, mirror_selfie={mirror_selfie_mode}, ref {ref_pil.width}×{ref_pil.height})"
        )

        max_retries = 3
        last_error  = None

        for attempt in range(1, max_retries + 1):
            try:
                with temporary_env_var("HTTP_PROXY", proxy), \
                     temporary_env_var("HTTPS_PROXY", proxy):
                    response = model_instance.generate_content(
                        [prompt, ref_pil],
                        generation_config=generation_config,
                    )
                raw_text  = response.text.strip()
                pose_sets = _parse_pose_sets(raw_text, pose_count)
                raw_out   = "\n---\n".join(pose_sets)

                logger.info(f"✓ Parsed {len(pose_sets)} pose sets (attempt {attempt})")
                for i, ps in enumerate(pose_sets):
                    logger.debug(f"  [{i+1}] {ps[:80].replace(chr(10),' | ')}")

                # pose_sets is a Python list → OUTPUT_IS_LIST unpacks it downstream
                return (pose_sets, raw_out)

            except Exception as exc:
                last_error = exc
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    time.sleep(1.5)

        logger.error(f"All {max_retries} attempts failed: {last_error}", exc_info=True)
        fallback = [f"[Error] {last_error}"]
        return (fallback, str(last_error))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiCarouselPoseInventor": GeminiCarouselPoseInventor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiCarouselPoseInventor": "Gemini Carousel Pose Inventor",
}
