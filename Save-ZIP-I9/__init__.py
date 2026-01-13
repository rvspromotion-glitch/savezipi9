from .adv_image_save import AdvancedImageSave

NODE_CLASS_MAPPINGS = {
    "AdvancedImageSave": AdvancedImageSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedImageSave": "Save Image (+ Zip Download)"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
