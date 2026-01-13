import os
from contextlib import contextmanager
from typing import Generator

import torch
from PIL import Image


def images_to_pillow(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert ComfyUI image tensor(s) to PIL Image(s).
    
    Args:
        images: Tensor of shape [B, H, W, C] where B is batch size
        
    Returns:
        List of PIL Images
    """
    pil_images = []
    
    # Handle batch of images
    for i in range(images.shape[0]):
        img_tensor = images[i]
        
        # Convert from [H, W, C] tensor (0-1 float) to PIL Image
        # ComfyUI uses 0-1 range, convert to 0-255
        img_array = (img_tensor.cpu().numpy() * 255).astype('uint8')
        pil_img = Image.fromarray(img_array)
        pil_images.append(pil_img)
    
    return pil_images


@contextmanager
def temporary_env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """
    Temporarily set an environment variable.
    
    Args:
        key: Environment variable name
        value: Value to set (or None to skip)
        
    Yields:
        None
    """
    if value is None:
        yield
        return
    
    old_value = os.environ.get(key)
    
    try:
        os.environ[key] = value
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value
