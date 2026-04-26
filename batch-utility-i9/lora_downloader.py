"""
LoraDownloader
==============
Paste one URL per line → downloads all LoRAs → outputs a list of
lora_name strings that wire directly into Multi LoRA Loader.

On re-run each file is checked first; already-present files are skipped
instantly with zero download time.

Download priority per URL:
  1. gdown  — Google Drive URLs (uses file ID extracted from the link)
  2. wget   — everything else
  3. curl   — fallback if wget missing
  4. requests streaming — last resort
"""

import logging
import os
import re
import shutil
import subprocess

import requests
import folder_paths

logger = logging.getLogger("LoraDownloader")

_CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_gdrive(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def _gdrive_file_id(url: str) -> str | None:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _infer_filename(url: str, index: int) -> str:
    clean = url.split("?")[0].rstrip("/")
    name  = clean.split("/")[-1]
    if name and "." in name:
        return name
    return f"lora_{index + 1}.safetensors"


def _run(cmd: list) -> None:
    subprocess.run(cmd, check=True)


def _download_url(url: str, tmp_path: str) -> None:
    if _is_gdrive(url) and shutil.which("gdown"):
        file_id = _gdrive_file_id(url)
        target  = file_id if file_id else url
        logger.info(f"  gdown {target[:60]}")
        _run(["gdown", target, "-O", tmp_path])

    elif shutil.which("wget"):
        logger.info(f"  wget {url[:80]}")
        _run(["wget", "-q", "--show-progress", "-O", tmp_path, url])

    elif shutil.which("curl"):
        logger.info(f"  curl {url[:80]}")
        _run(["curl", "-L", "--progress-bar", "-o", tmp_path, url])

    else:
        logger.info(f"  requests {url[:80]}")
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
                            logger.info(f"    {done/1024/1024:.0f}/{total/1024/1024:.0f} MB")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LoraDownloader:
    """
    Multi-URL LoRA downloader.  Paste one URL per line.
    Skips files that are already in the loras folder.
    Outputs a STRING list compatible with Multi LoRA Loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "One download URL per line. Google Drive share links supported.",
                }),
            },
            "optional": {
                "filenames": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   (
                        "One filename per line, matched by position "
                        "(e.g. my_lora.safetensors). Leave blank to infer from URL. "
                        "Required for Google Drive links where the name can't be guessed."
                    ),
                }),
                "subfolder": ("STRING", {
                    "default":   "",
                    "multiline": False,
                    "tooltip":   "Optional subfolder inside the loras directory.",
                }),
                "civitai_token": ("STRING", {
                    "default":   "",
                    "multiline": False,
                    "tooltip":   "CivitAI API token (appended as ?token= for civitai.com URLs).",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Strength passed along to Multi LoRA Loader for all LoRAs.",
                }),
            },
        }

    RETURN_TYPES   = ("STRING", "FLOAT")
    RETURN_NAMES   = ("lora_names", "strength")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION       = "download_loras"
    CATEGORY       = "loaders"

    def download_loras(
        self,
        urls: str,
        filenames: str = "",
        subfolder: str = "",
        civitai_token: str = "",
        strength: float = 1.0,
    ) -> tuple:
        # Defensive unwrap
        if isinstance(urls, list):          urls          = urls[0]          if urls          else ""
        if isinstance(filenames, list):     filenames     = filenames[0]     if filenames     else ""
        if isinstance(subfolder, list):     subfolder     = subfolder[0]     if subfolder     else ""
        if isinstance(civitai_token, list): civitai_token = civitai_token[0] if civitai_token else ""
        if isinstance(strength, list):      strength      = strength[0]      if strength      else 1.0

        url_list  = [u.strip() for u in (urls      or "").splitlines() if u.strip()]
        name_list = [n.strip() for n in (filenames or "").splitlines() if n.strip()]

        if not url_list:
            raise ValueError("LoraDownloader: no URLs provided")

        loras_base = folder_paths.get_folder_paths("loras")[0]
        dest_dir   = os.path.join(loras_base, subfolder) if subfolder else loras_base
        os.makedirs(dest_dir, exist_ok=True)

        results = []

        for idx, url in enumerate(url_list):
            # Append CivitAI token if needed
            if civitai_token and "civitai.com" in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}token={civitai_token}"

            filename  = name_list[idx] if idx < len(name_list) else _infer_filename(url, idx)
            dest_path = os.path.join(dest_dir, filename)
            lora_name = (os.path.join(subfolder, filename) if subfolder else filename).replace("\\", "/")

            if os.path.exists(dest_path):
                size_mb = os.path.getsize(dest_path) / 1024 / 1024
                logger.info(f"[{idx+1}/{len(url_list)}] ✓ Already present: {lora_name} ({size_mb:.1f} MB)")
                results.append(lora_name)
                continue

            logger.info(f"[{idx+1}/{len(url_list)}] Downloading → {filename}")
            tmp_path = dest_path + ".tmp"
            try:
                _download_url(url, tmp_path)
                os.replace(tmp_path, dest_path)
                size_mb = os.path.getsize(dest_path) / 1024 / 1024
                logger.info(f"[{idx+1}/{len(url_list)}] ✓ Done: {lora_name} ({size_mb:.1f} MB)")
                results.append(lora_name)
            except Exception as exc:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except OSError: pass
                raise RuntimeError(f"Download failed for {filename}: {exc}") from exc

        return (results, strength)


NODE_CLASS_MAPPINGS       = {"LoraDownloader": LoraDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraDownloader": "LoRA Downloader"}
