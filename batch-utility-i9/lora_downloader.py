"""
LoraDownloader
==============
Downloads a LoRA .safetensors file into ComfyUI's loras folder.

- Uses gdown for Google Drive URLs (handles auth/confirmation automatically).
- Uses wget for everything else.
- Falls back to requests streaming if neither CLI tool is available.
- Instant passthrough if the file already exists — zero overhead on re-runs.
- Output lora_name feeds directly into Multi LoRA Loader.
"""

import logging
import os
import shutil
import subprocess

import requests
import folder_paths

logger = logging.getLogger("LoraDownloader")

_CHUNK = 8 * 1024 * 1024


def _is_gdrive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _download(url: str, tmp_path: str) -> None:
    """Download url → tmp_path using the best available tool."""

    if _is_gdrive(url) and shutil.which("gdown"):
        logger.info("Using gdown (Google Drive)")
        subprocess.run(
            ["gdown", "--fuzzy", url, "-O", tmp_path],
            check=True,
        )

    elif shutil.which("wget"):
        logger.info("Using wget")
        subprocess.run(
            ["wget", "-O", tmp_path, url],
            check=True,
        )

    elif shutil.which("curl"):
        logger.info("Using curl")
        subprocess.run(
            ["curl", "-L", "-o", tmp_path, url],
            check=True,
        )

    else:
        logger.info("Falling back to requests streaming")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done  = 0
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            logger.info(f"  {done/1024/1024:.0f}/{total/1024/1024:.0f} MB")


class LoraDownloader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Direct download URL or Google Drive share link.",
                }),
                "filename": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Filename to save as. Leave empty to infer from URL.",
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
                    "tooltip": "CivitAI API token (appended as ?token= for CivitAI URLs).",
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("lora_name",)
    FUNCTION      = "download_lora"
    CATEGORY      = "loaders"

    def download_lora(self, url, filename="", subfolder="", civitai_token="") -> tuple:
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

        # Append CivitAI token if provided
        if civitai_token and "civitai.com" in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}token={civitai_token}"

        # Infer filename from URL
        if not filename:
            clean = url.split("?")[0].rstrip("/")
            filename = clean.split("/")[-1] or "lora.safetensors"
            if "." not in filename:
                filename += ".safetensors"

        loras_base = folder_paths.get_folder_paths("loras")[0]
        dest_dir   = os.path.join(loras_base, subfolder) if subfolder else loras_base
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, filename)
        lora_name = (os.path.join(subfolder, filename) if subfolder else filename).replace("\\", "/")

        # Passthrough if already present
        if os.path.exists(dest_path):
            logger.info(f"✓ Already exists, skipping: {lora_name} ({os.path.getsize(dest_path)/1024/1024:.1f} MB)")
            return (lora_name,)

        logger.info(f"Downloading → {filename}")
        tmp_path = dest_path + ".tmp"
        try:
            _download(url, tmp_path)
            os.replace(tmp_path, dest_path)
            logger.info(f"✓ Done: {lora_name} ({os.path.getsize(dest_path)/1024/1024:.1f} MB)")
        except Exception as exc:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass
            raise RuntimeError(f"LoRA download failed: {exc}") from exc

        return (lora_name,)


NODE_CLASS_MAPPINGS      = {"LoraDownloader": LoraDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraDownloader": "LoRA Downloader"}
