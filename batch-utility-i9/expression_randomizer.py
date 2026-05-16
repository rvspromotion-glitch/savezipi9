"""
ExpressionRandomizerBatch
=========================
Injects a randomly selected facial expression description into each prompt
in a batch.  Filters narrow the expression pool before sampling so every
image gets an expression that fits the desired mood, mouth position, energy
level, and hand/mirror requirements.

The same seed produces the same per-image selection sequence across runs.

Filter inputs
-------------
mood_filter   : comma-separated moods, or "all"
                options: joyful, intense, confident, warm, soft, neutral,
                         flirty, playful, attitude
mouth_filter  : comma-separated mouth types, or "all"
                options: open, pout, smile, closed, tongue out, parted, bite
energy_filter : comma-separated energy levels, or "all"
                options: high, medium, low
hands         : "any" | "with hands" | "no hands"
mirror        : "any" | "mirror only" | "no mirror"
source        : "all" | "real only" | "ai only"
"""

import logging
import random

logger = logging.getLogger("ExpressionRandomizerBatch")

# ---------------------------------------------------------------------------
# Expression library
# ---------------------------------------------------------------------------

_EXPRESSIONS = [
    {
        "title": "Genuine laugh",
        "desc": "Open mouth laugh, head slightly back, eyes nearly closed from cheek lift, fully uninhibited",
        "mood": "joyful",
        "mouth": "open",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Hype shout",
        "desc": "Mouth wide open in a raw explosive shout, one eye squinting from force, playful energy underneath",
        "mood": "intense",
        "mouth": "open",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Baddie unbothered",
        "desc": "Lips slightly pouted, jaw relaxed, eyes hidden or half-lidded, peace sign held up close to face at chin/cheek level",
        "mood": "confident",
        "mouth": "pout",
        "energy": "low",
        "hands": True,
        "hand_position": "peace sign at chin/cheek level",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Warm confident smile",
        "desc": "Soft closed-cheek smile, eyes slightly creased, relaxed and approachable",
        "mood": "warm",
        "mouth": "smile",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Soft distracted smile",
        "desc": "Slight lip upturn while gazing off to the side, gentle and natural, not performing for camera",
        "mood": "soft",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Neutral direct gaze",
        "desc": "Lips closed and relaxed, eyes steady into camera, quietly commanding, no warmth performed",
        "mood": "neutral",
        "mouth": "closed",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Cute wink",
        "desc": "One eye closed in deliberate wink, wide open smile, flirty and self-aware",
        "mood": "flirty",
        "mouth": "smile",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Coy smirk",
        "desc": "Small asymmetric smile, one eye doing more work than the other, knowing and calculated sweetness",
        "mood": "flirty",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Sultry squint pout",
        "desc": "One eye squinting more than the other, lips pushed forward in a soft pout, intense direct gaze into camera",
        "mood": "intense",
        "mouth": "pout",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Tongue out playful",
        "desc": "Mouth wide open, tongue out, one eye squinting, high energy and cheeky, self-aware",
        "mood": "playful",
        "mouth": "tongue out",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Cute pout with finger kiss",
        "desc": "Lips pushed into a soft pout, one eye slightly more closed, fingers pinched together pointing upward held at chin/lip level in a delicate mwah gesture",
        "mood": "flirty",
        "mouth": "pout",
        "energy": "low",
        "hands": True,
        "hand_position": "fingers pinched at chin/lip level",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Doe-eyed resting on hand",
        "desc": "Wide open eyes with soft vacant stare, head resting sideways into open palm at cheek level, lips slightly parted and relaxed, dreamy and gentle",
        "mood": "soft",
        "mouth": "parted",
        "energy": "low",
        "hands": True,
        "hand_position": "open palm cradling cheek",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Double peace sign grin",
        "desc": "Wide genuine smile, both hands raised with peace signs framing the face at eye level on either side, eyes squinting from the smile",
        "mood": "joyful",
        "mouth": "smile",
        "energy": "high",
        "hands": True,
        "hand_position": "both hands framing face at eye level",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Laughing behind hand",
        "desc": "Caught mid-laugh, one hand raised covering most of the face, smile breaking through underneath, shy unguarded energy",
        "mood": "playful",
        "mouth": "smile",
        "energy": "medium",
        "hands": True,
        "hand_position": "hand covering face",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Tongue out eyes closed",
        "desc": "Eyes fully closed, tongue extended downward, lips parted wide, nose slightly scrunched, playful and cheeky",
        "mood": "playful",
        "mouth": "tongue out",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Open mouth joy shout",
        "desc": "Mouth wide open mid-shout or singing along, eyes squinted, head slightly back, nose wrinkling from intensity, warm and uninhibited",
        "mood": "joyful",
        "mouth": "open",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Both hands prayer-cupped at cheek",
        "desc": "Soft neutral expression, both hands brought together resting against the side of the face fingers pointing upward, gentle and demure",
        "mood": "soft",
        "mouth": "closed",
        "energy": "low",
        "hands": True,
        "hand_position": "both hands prayer-cupped at cheek",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "One arm raised celebration",
        "desc": "Big open smile, one arm fully extended straight up above the head, body loose and open, pure celebratory energy",
        "mood": "joyful",
        "mouth": "smile",
        "energy": "high",
        "hands": True,
        "hand_position": "one arm fully raised above head",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Skeptical squint smirk",
        "desc": "One eye squinting more than the other, lips pressed into slight asymmetric smirk, nose slightly scrunched, unimpressed or suspicious attitude",
        "mood": "attitude",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Soft smile with cheek peace sign",
        "desc": "Gentle closed-mouth smile, cheeks slightly pushed up, one hand resting against cheek and the other holding a peace sign at face level beside it",
        "mood": "warm",
        "mouth": "smile",
        "energy": "low",
        "hands": True,
        "hand_position": "one hand on cheek, one peace sign at face level",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Eyes down lip bite",
        "desc": "Gaze directed downward, lower lip subtly caught between the teeth, introspective and slightly tense energy",
        "mood": "soft",
        "mouth": "bite",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Both hands in hair looking away",
        "desc": "Neutral to slightly serious expression, both hands raised holding hair above the head, gaze directed off to the side, editorial and confident",
        "mood": "confident",
        "mouth": "closed",
        "energy": "medium",
        "hands": True,
        "hand_position": "both hands raised holding hair above head",
        "mirror": False,
        "source": "real",
    },
    {
        "title": "Mirror selfie pout",
        "desc": "Lips pushed into a deliberate pout, gaze directed at phone in hand, body angled sideways. Mirror selfie.",
        "mood": "attitude",
        "mouth": "pout",
        "energy": "low",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "real",
    },
    {
        "title": "Mirror selfie tongue out",
        "desc": "Tongue extended out, eyes looking at phone in hand, cheeky and casual. Mirror selfie.",
        "mood": "playful",
        "mouth": "tongue out",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "real",
    },
    {
        "title": "Surprised lips parted",
        "desc": "Lips parted in a soft O shape, eyes wide, brows slightly raised, caught-off-guard or mock surprise. Mirror selfie.",
        "mood": "neutral",
        "mouth": "parted",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "real",
    },
    {
        "title": "Sideways smirk at phone",
        "desc": "Small asymmetric smirk, gaze directed down at phone in hand, nonchalant and slightly amused. Mirror selfie.",
        "mood": "attitude",
        "mouth": "smile",
        "energy": "low",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "real",
    },
    {
        "title": "Puffed cheeks pout",
        "desc": "Cheeks puffed with air, lips pushed forward into a pout simultaneously, silly and playful. Mirror selfie.",
        "mood": "playful",
        "mouth": "pout",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "real",
    },
    {
        "title": "Chin tilt smirk",
        "desc": "Chin slightly lifted upward, lips curled into a light one-sided smirk, gaze looking slightly downward into camera with quiet superiority",
        "mood": "confident",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Biting finger tip",
        "desc": "One finger brought up to the mouth with the tip gently between the teeth, eyes wide and soft, lips slightly parted around it, coy and teasing",
        "mood": "flirty",
        "mouth": "bite",
        "energy": "low",
        "hands": True,
        "hand_position": "one finger raised to lips",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Over shoulder glance",
        "desc": "Head turned looking back over one shoulder, lips closed in a subtle soft smile, eyes catching the camera mid-turn, effortlessly alluring",
        "mood": "flirty",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Closed eye serene smile",
        "desc": "Eyes fully closed, soft genuine smile, head tilted slightly, expression of pure contentment and calm",
        "mood": "soft",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Hand framing jaw",
        "desc": "One hand brought up with fingers loosely curled under the jaw, tilting the face slightly upward, lips relaxed and soft, editorial and composed",
        "mood": "neutral",
        "mouth": "closed",
        "energy": "low",
        "hands": True,
        "hand_position": "fingers loosely framing jaw from below",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Teeth bared fierce",
        "desc": "Both rows of teeth fully visible in a wide intense grin, eyes sharp and narrowed, high energy and commanding, not warm — fierce",
        "mood": "intense",
        "mouth": "open",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Nose scrunch grin",
        "desc": "Full wide smile causing the nose to scrunch upward, eyes crinkled, cheeks high and full, authentically cute and unguarded",
        "mood": "joyful",
        "mouth": "smile",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Pouty lip pull down",
        "desc": "Bottom lip pulled down slightly with fingertip, exposing bottom teeth edge, eyes direct and teasing into camera",
        "mood": "flirty",
        "mouth": "pout",
        "energy": "medium",
        "hands": True,
        "hand_position": "one fingertip pulling lightly at lower lip",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Thinking face look up",
        "desc": "Eyes directed upward and to one side, lips pressed lightly together or slightly pursed, one finger raised to the temple, mock-contemplative and playful",
        "mood": "playful",
        "mouth": "closed",
        "energy": "low",
        "hands": True,
        "hand_position": "one finger raised to temple",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Whispering lean in",
        "desc": "Head tilted slightly forward, one hand raised to the side of the mouth as if shielding a whisper, eyes wide and conspiratorial",
        "mood": "playful",
        "mouth": "parted",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand cupped at side of mouth",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Blowing a kiss",
        "desc": "Lips pursed forward mid-blow, one hand raised flat in front of the mouth fingers together as if sending the kiss forward, eyes soft and warm",
        "mood": "flirty",
        "mouth": "pout",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand raised flat in front of mouth",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Dead stare smirk",
        "desc": "Completely flat expressionless eyes, no brow movement, small barely-there smirk at one corner of the mouth, deeply dry and deadpan",
        "mood": "attitude",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Shy look away cover mouth",
        "desc": "Gaze directed off to the side or downward, one hand raised loosely covering the lower half of the face at mouth level, smile hidden behind it",
        "mood": "soft",
        "mouth": "smile",
        "energy": "low",
        "hands": True,
        "hand_position": "one hand loosely covering mouth",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Double hand frame face",
        "desc": "Both hands raised on either side of the face with palms forward and fingers spread, framing the face, wide eyes and open expression",
        "mood": "playful",
        "mouth": "parted",
        "energy": "high",
        "hands": True,
        "hand_position": "both palms framing face at either side",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Soft upper lip bite",
        "desc": "Upper lip caught lightly between the teeth rather than lower, gaze steady and direct into camera, subtle and intentional",
        "mood": "flirty",
        "mouth": "bite",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Eyes closed tongue out lean",
        "desc": "Eyes closed, tongue extended to one side of the mouth rather than straight out, head tilted slightly in the same direction, loose and goofy",
        "mood": "playful",
        "mouth": "tongue out",
        "energy": "medium",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Forehead rest on hands",
        "desc": "Both hands stacked or laced together with forehead resting down onto them, eyes looking up at camera from below, soft and intimate",
        "mood": "soft",
        "mouth": "closed",
        "energy": "low",
        "hands": True,
        "hand_position": "both hands stacked under forehead",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Smug side eye",
        "desc": "Head facing mostly forward but eyes cut sharply to one side, one brow slightly raised, lips closed in a barely suppressed smirk",
        "mood": "attitude",
        "mouth": "smile",
        "energy": "low",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Open mouth shock grin",
        "desc": "Mouth dropped wide open in exaggerated mock shock, eyes wide and brows raised high, caught mid-reaction, fully performative",
        "mood": "playful",
        "mouth": "open",
        "energy": "high",
        "hands": False,
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Hand on chest sincere",
        "desc": "One hand placed flat against the chest, soft open smile, eyes warm and direct, genuine and heartfelt energy",
        "mood": "warm",
        "mouth": "smile",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand flat on chest",
        "mirror": False,
        "source": "ai",
    },
    {
        "title": "Mirror selfie eyes closed pout",
        "desc": "Eyes fully closed, lips pushed into a deliberate pout, free hand resting loosely at side. Mirror selfie.",
        "mood": "flirty",
        "mouth": "pout",
        "energy": "low",
        "hands": True,
        "hand_position": "one hand holding phone toward mirror",
        "mirror": True,
        "source": "ai",
    },
    {
        "title": "Two finger gun point",
        "desc": "One hand raised with index and middle finger extended pointing forward like a gun toward camera, other hand on hip, lips in a confident smirk",
        "mood": "confident",
        "mouth": "smile",
        "energy": "medium",
        "hands": True,
        "hand_position": "one hand finger-gun pointing toward camera at chest level",
        "mirror": False,
        "source": "ai",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_filter(raw: str) -> set[str] | None:
    """Return a set of allowed values, or None meaning 'no filter'."""
    if isinstance(raw, list):
        raw = raw[0] if raw else "all"
    raw = (raw or "all").strip().lower()
    if raw in ("all", "", "*"):
        return None
    return {v.strip() for v in raw.split(",") if v.strip()}


def _filter_expressions(
    mood_filter,
    mouth_filter,
    energy_filter,
    hands: str,
    mirror: str,
    source: str,
) -> list[dict]:
    moods    = _parse_filter(mood_filter)
    mouths   = _parse_filter(mouth_filter)
    energies = _parse_filter(energy_filter)

    if isinstance(hands, list):
        hands = hands[0] if hands else "any"
    if isinstance(mirror, list):
        mirror = mirror[0] if mirror else "any"
    if isinstance(source, list):
        source = source[0] if source else "all"

    pool = []
    for expr in _EXPRESSIONS:
        if moods    and expr["mood"]   not in moods:
            continue
        if mouths   and expr["mouth"]  not in mouths:
            continue
        if energies and expr["energy"] not in energies:
            continue
        if hands == "with hands" and not expr["hands"]:
            continue
        if hands == "no hands" and expr["hands"]:
            continue
        if mirror == "mirror only" and not expr["mirror"]:
            continue
        if mirror == "no mirror" and expr["mirror"]:
            continue
        if source == "real only" and expr["source"] != "real":
            continue
        if source == "ai only" and expr["source"] != "ai":
            continue
        pool.append(expr)

    return pool


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ExpressionRandomizerBatch:
    """
    Randomly injects an expression description into each prompt in a batch.
    Filters control which expressions are eligible before sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31, "step": 1,
                    "tooltip": "Controls the per-image selection sequence. Same seed = same expressions.",
                }),
            },
            "optional": {
                "mood_filter": ("STRING", {
                    "default": "all",
                    "tooltip": (
                        "Comma-separated moods to allow, or 'all'.\n"
                        "Options: joyful, intense, confident, warm, soft, "
                        "neutral, flirty, playful, attitude"
                    ),
                }),
                "mouth_filter": ("STRING", {
                    "default": "all",
                    "tooltip": (
                        "Comma-separated mouth types to allow, or 'all'.\n"
                        "Options: open, pout, smile, closed, tongue out, parted, bite"
                    ),
                }),
                "energy_filter": ("STRING", {
                    "default": "all",
                    "tooltip": (
                        "Comma-separated energy levels to allow, or 'all'.\n"
                        "Options: high, medium, low"
                    ),
                }),
                "hands": (
                    ["any", "with hands", "no hands"],
                    {"default": "any"},
                ),
                "mirror": (
                    ["any", "mirror only", "no mirror"],
                    {"default": "any"},
                ),
                "source": (
                    ["all", "real only", "ai only"],
                    {"default": "all"},
                ),
                "include_title": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Prepend the expression title before its description in the output.",
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "tooltip": "Glue string inserted between the original prompt and the expression text.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("prompts",)
    FUNCTION      = "apply"
    CATEGORY      = "Gemini/Creative"

    INPUT_IS_LIST  = (True, False, False, False, False, False, False, False, False, False)
    OUTPUT_IS_LIST = (True,)

    def apply(
        self,
        prompts: list[str],
        seed: int = 0,
        mood_filter="all",
        mouth_filter="all",
        energy_filter="all",
        hands="any",
        mirror="any",
        source="all",
        include_title: bool = False,
        separator=", ",
    ) -> tuple:
        # Unwrap scalars that ComfyUI may wrap in lists
        if isinstance(seed, list):
            seed = seed[0] if seed else 0
        if isinstance(include_title, list):
            include_title = include_title[0] if include_title else False
        if isinstance(separator, list):
            separator = separator[0] if separator else ", "

        pool = _filter_expressions(mood_filter, mouth_filter, energy_filter, hands, mirror, source)

        if not pool:
            logger.warning(
                "ExpressionRandomizerBatch: no expressions match the current filters — "
                "returning prompts unchanged."
            )
            return (list(prompts),)

        logger.info(
            f"ExpressionRandomizerBatch: {len(pool)} expressions in pool, "
            f"{len(prompts)} prompts, seed={seed}"
        )

        rng = random.Random(seed)
        results = []

        for idx, prompt in enumerate(prompts):
            expr = rng.choice(pool)
            if include_title:
                expr_text = f"{expr['title']}: {expr['desc']}"
            else:
                expr_text = expr["desc"]

            combined = f"{prompt}{separator}{expr_text}"
            results.append(combined)
            logger.debug(f"  [{idx}] {expr['title']} → {combined[:100]}")

        logger.info(f"✓ ExpressionRandomizerBatch: processed {len(results)} prompts")
        return (results,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ExpressionRandomizerBatch": ExpressionRandomizerBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpressionRandomizerBatch": "Expression Randomizer (Batch)",
}
