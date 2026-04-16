import io
import logging
import os
import zipfile

import folder_paths
import numpy as np
from aiohttp import web
from PIL import Image
from server import PromptServer

logger = logging.getLogger("DatasetSave")


# ---------------------------------------------------------------------------
# Node 1 – BatchImageRenamer
# ---------------------------------------------------------------------------

class BatchImageRenamer:
    """
    Splits a batched image tensor into a list of individual images and assigns
    each one a sequential filename stem (1, 2, 3 … or prefix_1, prefix_2 …).

    Connect:
      images  ← any IMAGE batch (e.g. from Load Image Batch)
      ↓
      images  → DatasetCaptionSave  (list of [1,H,W,C] tensors)
      filenames → DatasetCaptionSave (list of strings: "1", "2", …)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "start_index": ("INT", {
                    "default": 1, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "First number in the sequence.",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "placeholder": "e.g. 'photo_'  →  photo_1, photo_2 …",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "filenames")
    FUNCTION = "rename"
    CATEGORY = "image/dataset"
    OUTPUT_IS_LIST = (True, True)

    def rename(self, images, start_index=1, prefix=""):
        # Defensive unwrap (ComfyUI may wrap widget values in lists)
        if isinstance(start_index, list):
            start_index = start_index[0] if start_index else 1
        if isinstance(prefix, list):
            prefix = prefix[0] if prefix else ""
        prefix = prefix or ""

        batch_size = images.shape[0]
        image_list = []
        filename_list = []

        for i in range(batch_size):
            image_list.append(images[i:i+1])           # keep [1,H,W,C] shape
            filename_list.append(f"{prefix}{start_index + i}")

        logger.info(
            f"✓ BatchImageRenamer: {batch_size} images → "
            f"{filename_list[0]} … {filename_list[-1]}"
        )
        return (image_list, filename_list)


# ---------------------------------------------------------------------------
# Node 2 – DatasetCaptionSave
# ---------------------------------------------------------------------------

class DatasetCaptionSave:
    """
    Saves paired image + caption files ready for training datasets.

    Typical wiring
    --------------
    BatchImageRenamer.images    → images
    BatchImageRenamer.filenames → filenames
    GeminiBatchNode.prompts     → captions

    Result on disk
    --------------
    <output>/dataset/
        1.png   1.txt
        2.png   2.txt
        …

    The "Download Dataset ZIP" button packages everything so the dataset is
    instantly ready to use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":    ("IMAGE",  {"forceInput": True}),
                "captions":  ("STRING", {"forceInput": True}),
                "filenames": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "output_subfolder": ("STRING", {
                    "default": "dataset",
                    "multiline": False,
                    "tooltip": "Subfolder inside ComfyUI's output directory.",
                }),
                "image_format": (["png", "jpg"],),
                "jpg_quality":  ("INT", {
                    "default": 95, "min": 1, "max": 100, "step": 1,
                    "tooltip": "JPEG quality (ignored when image_format = png).",
                }),
                "auto_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically download the ZIP when the workflow finishes.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_dataset"
    OUTPUT_NODE = True
    CATEGORY = "image/dataset"

    # Required inputs first, then optional – tuple must match that order
    INPUT_IS_LIST = (True, True, True, False, False, False, False)

    def save_dataset(
        self,
        images,
        captions,
        filenames,
        output_subfolder="dataset",
        image_format="png",
        jpg_quality=95,
        auto_download=False,
    ):
        # Defensive unwrap for scalar optional inputs
        if isinstance(output_subfolder, list):
            output_subfolder = output_subfolder[0] if output_subfolder else "dataset"
        if isinstance(image_format, list):
            image_format = image_format[0] if image_format else "png"
        if isinstance(jpg_quality, list):
            jpg_quality = jpg_quality[0] if jpg_quality else 95
        if isinstance(auto_download, list):
            auto_download = auto_download[0] if auto_download else False

        output_subfolder = output_subfolder or "dataset"

        output_dir = os.path.join(folder_paths.get_output_directory(), output_subfolder)
        os.makedirs(output_dir, exist_ok=True)

        count = min(len(images), len(captions), len(filenames))
        saved_files = []

        for idx in range(count):
            img_tensor = images[idx]    # [1, H, W, C]
            caption    = captions[idx]
            stem       = filenames[idx]

            # --- image ---
            arr = (img_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            if image_format == "jpg":
                img_name = f"{stem}.jpg"
                img_path = os.path.join(output_dir, img_name)
                pil.save(img_path, format="JPEG", quality=jpg_quality)
            else:
                img_name = f"{stem}.png"
                img_path = os.path.join(output_dir, img_name)
                pil.save(img_path, format="PNG")

            saved_files.append({"path": img_path, "filename": img_name})

            # --- caption txt ---
            txt_name = f"{stem}.txt"
            txt_path = os.path.join(output_dir, txt_name)
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(caption)

            saved_files.append({"path": txt_path, "filename": txt_name})
            logger.debug(f"  [{idx}] saved {img_name} + {txt_name}")

        logger.info(
            f"✓ DatasetCaptionSave: {count} pairs saved to {output_dir}"
        )

        return {
            "ui": {
                "saved_files":   saved_files,
                "dataset_count": [count],
                "output_dir":    [output_dir],
                "auto_download": [auto_download],
            }
        }


# ---------------------------------------------------------------------------
# Server route  –  /download_dataset_zip
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.post("/download_dataset_zip")
async def download_dataset_zip(request):
    _log = logging.getLogger("DatasetZipDownload")
    try:
        data     = await request.json()
        files    = data.get("files", [])
        zip_name = data.get("zip_name", "dataset.zip")

        if not files:
            return web.Response(status=400, text="No files provided")

        total = len(files)
        _log.info(f"Building dataset ZIP: {total} files …")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fi in files:
                path     = fi.get("path")
                arcname  = fi.get("filename")
                if path and os.path.exists(path):
                    zf.write(path, arcname=arcname)
                else:
                    _log.warning(f"Missing file: {path}")

        buf.seek(0)
        body = buf.read()
        _log.info(f"✓ ZIP ready: {total} files, {len(body)/1024/1024:.2f} MB")

        return web.Response(
            body=body,
            headers={
                "Content-Type":        "application/zip",
                "Content-Disposition": f'attachment; filename="{zip_name}"',
            },
        )
    except Exception as exc:
        _log.error(f"Dataset ZIP failed: {exc}", exc_info=True)
        return web.Response(status=500, text=str(exc))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BatchImageRenamer":   BatchImageRenamer,
    "DatasetCaptionSave":  DatasetCaptionSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageRenamer":   "Batch Image Renamer",
    "DatasetCaptionSave":  "Dataset Caption Save (+ ZIP)",
}
