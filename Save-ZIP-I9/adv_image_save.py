import os
import json
import zipfile
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
from server import PromptServer
from aiohttp import web

class AdvancedImageSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        
        # Handle tensor properly
        if hasattr(images, 'shape'):
            batch_size = images.shape[0]
            height = images.shape[1]
            width = images.shape[2]
        else:
            # Fallback if images is a list
            batch_size = len(images)
            height = images[0].shape[0] if hasattr(images[0], 'shape') else images[0].size[1]
            width = images[0].shape[1] if hasattr(images[0], 'shape') else images[0].size[0]
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, width, height
        )
        
        results = list()
        saved_files = []
        
        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()

            if not isinstance(extra_pnginfo, dict):
                extra_pnginfo = {}

            # Add workflow metadata to PNG
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            file = f"{filename}_{counter:05}_.png"
            img_path = os.path.join(full_output_folder, file)
            img.save(img_path, pnginfo=metadata, compress_level=self.compress_level)
            
            saved_files.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "path": img_path
            })
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            
            counter += 1

        # Store file paths for zip download
        return {"ui": {"images": results, "saved_files": saved_files}}

@PromptServer.instance.routes.post("/download_batch_zip")
async def download_batch_zip(request):
    try:
        data = await request.json()
        files = data.get("files", [])
        
        if not files:
            return web.Response(status=400, text="No files provided")
        
        # Create zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in files:
                file_path = file_info.get("path")
                filename = file_info.get("filename")
                
                if file_path and os.path.exists(file_path):
                    zip_file.write(file_path, arcname=filename)
        
        zip_buffer.seek(0)
        
        return web.Response(
            body=zip_buffer.read(),
            headers={
                'Content-Type': 'application/zip',
                'Content-Disposition': f'attachment; filename="comfyui_batch.zip"'
            }
        )
    except Exception as e:
        return web.Response(status=500, text=str(e))

NODE_CLASS_MAPPINGS = {
    "AdvancedImageSave": AdvancedImageSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedImageSave": "Save Image (+ Zip Download)"
}
