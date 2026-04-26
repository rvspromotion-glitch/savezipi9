"""
MultiLoraLoader
===============
Receives a list of lora_name strings from LoraDownloader and a strength
float, applies every LoRA to MODEL + CLIP, and passes them through.

Uses direct filesystem scan (not the startup cache) so LoRAs downloaded
during the current session are found immediately.
"""

import logging
import os

import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger("MultiLoraLoader")


def _find_lora_path(lora_name: str) -> str | None:
    """Scan loras directories directly — bypasses the startup file cache."""
    for base in folder_paths.get_folder_paths("loras"):
        candidate = os.path.join(base, lora_name)
        if os.path.isfile(candidate):
            return candidate
    return None


class MultiLoraLoader:
    """
    Loads all LoRAs from the lora_names list output of LoraDownloader.
    Wire:  LoraDownloader.lora_names → lora_names
           LoraDownloader.strength   → strength
           CheckpointLoader.MODEL    → model
           CheckpointLoader.CLIP     → clip
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "clip":       ("CLIP",),
                "lora_names": ("STRING", {"forceInput": True}),
                "strength":   ("FLOAT",  {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "forceInput": True,
                }),
            },
        }

    INPUT_IS_LIST  = (False, False, True, False)
    RETURN_TYPES   = ("MODEL", "CLIP")
    FUNCTION       = "load_loras"
    CATEGORY       = "loaders"

    def load_loras(self, model, clip, lora_names: list, strength=1.0) -> tuple:
        if isinstance(strength, list):
            strength = strength[0] if strength else 1.0

        for idx, lora_name in enumerate(lora_names):
            if not lora_name or not str(lora_name).strip():
                continue

            lora_name = str(lora_name).strip()
            lora_path = _find_lora_path(lora_name)

            if not lora_path:
                logger.warning(f"[{idx+1}] LoRA not found: {lora_name} — skipping")
                continue

            logger.info(f"[{idx+1}] Loading {lora_name}  strength={strength:.2f}")
            try:
                lora        = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
                logger.info(f"[{idx+1}] ✓ Applied {lora_name}")
            except Exception as exc:
                logger.error(f"[{idx+1}] Failed to load {lora_name}: {exc}", exc_info=True)
                raise RuntimeError(f"Multi LoRA Loader failed on {lora_name}: {exc}") from exc

        return (model, clip)


NODE_CLASS_MAPPINGS        = {"MultiLoraLoader": MultiLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiLoraLoader": "Multi LoRA Loader"}
