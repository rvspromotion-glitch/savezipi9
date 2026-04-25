"""
LoraDownloader
==============
Downloads a LoRA .safetensors file from a direct URL into ComfyUI's loras
folder before the rest of the workflow runs.

- Instant passthrough if the file already exists (no re-download).
- Streams in 8 MB chunks so large files don't block Python's GIL.
- Writes to a .tmp file first, then renames — no corrupt partials on crash.
- Output lora_name is the relative path expected by Load LoRA nodes
  (e.g. "my_lora.safetensors" or "subfolder/my_lora.safetensors").
"""

import logging
import os

import requests
import folder_paths

logger = logging.getLogger("LoraDownloader")

_CHUNK = 8 * 1024 * 1024   # 8 MB per chunk


class LoraDownloader:
    """
    Passthrough LoRA downloader.  Skips the download if the file already
    exists in the loras directory — zero overhead on subsequent runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Direct download URL to a .safetensors file.",
                }),
                "filename": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Filename to save as (e.g. my_lora.safetensors). "
                        "Leave empty to infer from the URL."
                    ),
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional subfolder inside the loras directory.",
                }),
                "civitai_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "CivitAI API token for models that require authentication.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("lora_name",)
    FUNCTION      = "download_lora"
    CATEGORY      = "loaders"

    # ------------------------------------------------------------------

    def download_lora(
        self,
        url: str,
        filename: str = "",
        subfolder: str = "",
        civitai_token: str = "",
    ) -> tuple:
        # Defensive unwrap
        if isinstance(url, list):           url           = url[0]           if url           else ""
        if isinstance(filename, list):      filename      = filename[0]      if filename      else ""
        if isinstance(subfolder, list):     subfolder     = subfolder[0]     if subfolder     else ""
        if isinstance(civitai_token, list): civitai_token = civitai_token[0] if civitai_token else ""

        url           = (url           or "").strip()
        filename      = (filename      or "").strip()
        subfolder     = (subfolder     or "").strip()
        civitai_token = (civitai_token or "").strip()

        if not url:
            raise ValueError("LoraDownloader: url is required")

        # Resolve filename from URL if not provided
        if not filename:
            filename = url.split("?")[0].rstrip("/").split("/")[-1]
            if not filename:
                filename = "lora.safetensors"
            elif "." not in filename:
                filename += ".safetensors"

        # Resolve destination directory
        loras_base = folder_paths.get_folder_paths("loras")[0]
        dest_dir   = os.path.join(loras_base, subfolder) if subfolder else loras_base
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, filename)
        lora_name = os.path.join(subfolder, filename).replace("\\", "/") if subfolder else filename

        # ── Passthrough: already downloaded ──────────────────────────────────
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / 1024 / 1024
            logger.info(f"✓ LoRA already present, skipping download: {lora_name} ({size_mb:.1f} MB)")
            return (lora_name,)

        # ── Download ──────────────────────────────────────────────────────────
        logger.info(f"Downloading LoRA → {lora_name}  ({url})")

        headers = {"Authorization": f"Bearer {civitai_token}"} if civitai_token else {}
        tmp_path = dest_path + ".tmp"

        try:
            with requests.get(url, stream=True, headers=headers, timeout=30) as r:
                r.raise_for_status()
                total    = int(r.headers.get("Content-Length", 0))
                total_mb = total / 1024 / 1024 if total else None
                downloaded = 0

                with open(tmp_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=_CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total_mb:
                            logger.info(
                                f"  {downloaded / 1024 / 1024:.0f} / {total_mb:.0f} MB  "
                                f"({downloaded / total * 100:.0f}%)"
                            )

            os.replace(tmp_path, dest_path)   # atomic on same filesystem
            final_mb = os.path.getsize(dest_path) / 1024 / 1024
            logger.info(f"✓ LoRA download complete: {lora_name} ({final_mb:.1f} MB)")

        except Exception as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(f"LoRA download failed: {exc}") from exc

        return (lora_name,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LoraDownloader": LoraDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraDownloader": "LoRA Downloader",
}
