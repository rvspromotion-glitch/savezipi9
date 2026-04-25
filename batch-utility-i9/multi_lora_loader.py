"""
MultiLoraLoader
===============
Loads up to 5 LoRAs from lora_name STRING inputs — wire LoraDownloader
nodes directly into lora_1 … lora_5.

Uses direct filesystem lookup instead of folder_paths.get_full_path so
LoRAs downloaded during this session (not in the startup cache) are found.
"""

import logging
import os

import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger("MultiLoraLoader")

_SLOTS = 5


def _find_lora_path(lora_name: str) -> str | None:
    """
    Locate a lora file by scanning loras directories directly.
    folder_paths.get_full_path uses a startup cache that misses files
    downloaded after ComfyUI launched.
    """
    for base in folder_paths.get_folder_paths("loras"):
        candidate = os.path.join(base, lora_name)
        if os.path.isfile(candidate):
            return candidate
    return None


class MultiLoraLoader:

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _SLOTS + 1):
            optional[f"lora_{i}"] = ("STRING", {"forceInput": True})
            optional[f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
        return {
            "required": {
                "model": ("MODEL",),
                "clip":  ("CLIP",),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION     = "load_loras"
    CATEGORY     = "loaders"

    def load_loras(self, model, clip, **kwargs):
        for i in range(1, _SLOTS + 1):
            lora_name = kwargs.get(f"lora_{i}")
            strength  = kwargs.get(f"strength_{i}", 1.0)

            if isinstance(lora_name, list):
                lora_name = lora_name[0] if lora_name else None
            if isinstance(strength, list):
                strength = strength[0] if strength else 1.0

            if not lora_name or not str(lora_name).strip():
                continue

            lora_name = str(lora_name).strip()
            lora_path = _find_lora_path(lora_name)

            if not lora_path:
                logger.warning(f"[slot {i}] LoRA file not found: {lora_name} — skipping")
                continue

            logger.info(f"[slot {i}] Loading {lora_name}  strength={strength:.2f}")
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
                logger.info(f"[slot {i}] ✓ Applied {lora_name}")
            except Exception as exc:
                logger.error(f"[slot {i}] Failed to load {lora_name}: {exc}", exc_info=True)
                raise RuntimeError(f"Multi LoRA Loader: failed on slot {i} ({lora_name}): {exc}") from exc

        return (model, clip)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MultiLoraLoader": MultiLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLoraLoader": "Multi LoRA Loader",
}
