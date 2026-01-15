"""
Batch-compatible version of NovaNodes from Image-Detection-Bypass-Utility
Original: https://github.com/PurinNyova/Image-Detection-Bypass-Utility

This wrapper adds batch processing capability while preserving all original
functionality. Processes each image in the batch individually through the
original pipeline.
"""

import torch
import numpy as np
from PIL import Image
import tempfile
import os
import logging

# Import the original processing function
try:
    from image_postprocess import process_image
    NOVA_AVAILABLE = True
except ImportError:
    NOVA_AVAILABLE = False
    print("WARNING: image_postprocess module not found. Install Image-Detection-Bypass-Utility first.")
    print("cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/PurinNyova/Image-Detection-Bypass-Utility")

def to_pil_from_any(img_input):
    """Convert torch tensor or numpy array to PIL RGB image."""
    if isinstance(img_input, torch.Tensor):
        img_np = img_input.cpu().numpy()
    elif isinstance(img_input, np.ndarray):
        img_np = img_input
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")

    # Handle different dimension orders
    if img_np.ndim == 4:
        img_np = img_np[0]  # Take first if batched
    if img_np.ndim == 3:
        if img_np.shape[0] in [1, 3, 4]:  # CHW format
            img_np = np.transpose(img_np, (1, 2, 0))
    elif img_np.ndim == 2:
        pass  # Grayscale
    else:
        raise ValueError(f"Unexpected image dimensions: {img_np.shape}")

    # Normalize to 0-255 uint8
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    return Image.fromarray(img_np).convert("RGB")


def _parse_int_list(s):
    """Parse comma-separated string to list of ints."""
    if isinstance(s, list):
        return s
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


def _parse_float_list(s):
    """Parse comma-separated string to list of floats."""
    if isinstance(s, list):
        return s
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",")]


class NovaNodesBatch:
    """
    Batch-compatible version of NovaNodes for AI detection bypass.
    Processes each image in the batch individually through the original pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                # Noise options
                "apply_noise_o": ("BOOLEAN", {"default": True}),
                "noise_std_frac": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.001}),

                # CLAHE options
                "apply_clahe": ("BOOLEAN", {"default": False}),
                "clahe_clip_limit": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "clahe_tile_size": ("INT", {"default": 8, "min": 2, "max": 64}),

                # FFT options
                "apply_fft": ("BOOLEAN", {"default": False}),
                "fft_cutoff": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_randomness": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_ref_image": ("IMAGE",),

                # GLCM options
                "apply_glcm": ("BOOLEAN", {"default": False}),
                "glcm_distances": ("STRING", {"default": "1,3"}),
                "glcm_angles": ("STRING", {"default": "0,45,90,135"}),
                "glcm_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # LBP options
                "apply_lbp": ("BOOLEAN", {"default": False}),
                "lbp_radius": ("INT", {"default": 1, "min": 1, "max": 8}),
                "lbp_n_points": ("INT", {"default": 8, "min": 4, "max": 24, "step": 4}),
                "lbp_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Perturbation options
                "apply_perturb": ("BOOLEAN", {"default": False}),
                "perturb_max_shift": ("INT", {"default": 2, "min": 1, "max": 10}),

                # Camera simulation options
                "apply_sim_camera": ("BOOLEAN", {"default": False}),
                "sim_bayer_pattern": ("BOOLEAN", {"default": True}),
                "sim_jpeg_cycles": ("INT", {"default": 1, "min": 0, "max": 5}),
                "sim_jpeg_quality": ("INT", {"default": 85, "min": 50, "max": 100}),
                "sim_vignetting": ("BOOLEAN", {"default": True}),
                "sim_chrom_aberration": ("BOOLEAN", {"default": False}),

                # AWB options
                "apply_awb": ("BOOLEAN", {"default": False}),
                "awb_ref_image": ("IMAGE",),

                # Non-semantic attack options
                "apply_non_semantic": ("BOOLEAN", {"default": False}),
                "ns_num_iter": ("INT", {"default": 10, "min": 1, "max": 100}),
                "ns_epsilon": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.1, "step": 0.001}),

                # LUT options
                "apply_lut": ("BOOLEAN", {"default": False}),
                "lut_path": ("STRING", {"default": ""}),

                # Blend options
                "apply_blend": ("BOOLEAN", {"default": False}),
                "blend_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_image": ("IMAGE",),

                # EXIF options
                "add_fake_exif": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "exif_data")
    FUNCTION = "process_batch"
    CATEGORY = "image/postprocessing"

    def process_batch(self, image, **kwargs):
        """Process entire batch of images through NovaNodes pipeline."""
        logger = logging.getLogger("NovaNodesBatch")

        if not NOVA_AVAILABLE:
            raise ImportError("image_postprocess module not found. Install Image-Detection-Bypass-Utility first.")

        # Get batch size
        batch_size = image.shape[0]
        logger.info(f"Processing batch of {batch_size} images through NovaNodes")

        # Process each image in the batch
        processed_tensors = []
        exif_data_list = []

        for idx in range(batch_size):
            logger.debug(f"Processing image {idx + 1}/{batch_size}")

            # Extract single image from batch
            single_image = image[idx:idx+1]  # Keep batch dimension

            # Process this image
            processed_tensor, exif_data = self._process_single(single_image, idx, **kwargs)

            processed_tensors.append(processed_tensor)
            exif_data_list.append(exif_data)

            if (idx + 1) % 10 == 0 or (idx + 1) == batch_size:
                logger.info(f"Processed {idx + 1}/{batch_size} images ({(idx + 1) / batch_size * 100:.1f}%)")

        # Concatenate all processed images into batch
        batch_output = torch.cat(processed_tensors, dim=0)

        # Combine EXIF data
        combined_exif = "\n---\n".join(exif_data_list)

        logger.info(f"✓ Batch processing complete: {batch_size} images, output shape {batch_output.shape}")

        return (batch_output, combined_exif)

    def _process_single(self, image, idx, **kwargs):
        """Process a single image through the NovaNodes pipeline."""
        # Convert tensor to PIL
        pil_img = to_pil_from_any(image[0])

        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            input_path = tmp_in.name
            pil_img.save(input_path, "PNG")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            # Build args namespace for process_image
            class Args:
                pass

            args = Args()

            # Noise
            args.noise = kwargs.get("apply_noise_o", True)
            args.noise_std_frac = kwargs.get("noise_std_frac", 0.02)

            # CLAHE
            args.clahe = kwargs.get("apply_clahe", False)
            args.clahe_clip_limit = kwargs.get("clahe_clip_limit", 2.0)
            args.clahe_tile_size = kwargs.get("clahe_tile_size", 8)

            # FFT
            args.fft = kwargs.get("apply_fft", False)
            args.fft_cutoff = kwargs.get("fft_cutoff", 0.1)
            args.fft_strength = kwargs.get("fft_strength", 0.3)
            args.fft_randomness = kwargs.get("fft_randomness", 0.1)

            # Handle FFT reference image
            fft_ref_image = kwargs.get("fft_ref_image")
            if fft_ref_image is not None and args.fft:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fft:
                    fft_ref_path = tmp_fft.name
                    fft_ref_pil = to_pil_from_any(fft_ref_image[0])
                    fft_ref_pil.save(fft_ref_path, "PNG")
                    args.fft_ref = fft_ref_path
            else:
                args.fft_ref = None

            # GLCM
            args.glcm = kwargs.get("apply_glcm", False)
            args.glcm_distances = _parse_int_list(kwargs.get("glcm_distances", "1,3"))
            args.glcm_angles = _parse_int_list(kwargs.get("glcm_angles", "0,45,90,135"))
            args.glcm_strength = kwargs.get("glcm_strength", 0.5)

            # LBP
            args.lbp = kwargs.get("apply_lbp", False)
            args.lbp_radius = kwargs.get("lbp_radius", 1)
            args.lbp_n_points = kwargs.get("lbp_n_points", 8)
            args.lbp_strength = kwargs.get("lbp_strength", 0.3)

            # Perturbation
            args.perturb = kwargs.get("apply_perturb", False)
            args.perturb_max_shift = kwargs.get("perturb_max_shift", 2)

            # Camera simulation
            args.sim_camera = kwargs.get("apply_sim_camera", False)
            args.sim_bayer_pattern = kwargs.get("sim_bayer_pattern", True)
            args.sim_jpeg_cycles = kwargs.get("sim_jpeg_cycles", 1)
            args.sim_jpeg_quality = kwargs.get("sim_jpeg_quality", 85)
            args.sim_vignetting = kwargs.get("sim_vignetting", True)
            args.sim_chrom_aberration = kwargs.get("sim_chrom_aberration", False)

            # AWB
            args.awb = kwargs.get("apply_awb", False)
            awb_ref_image = kwargs.get("awb_ref_image")
            if awb_ref_image is not None and args.awb:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_awb:
                    awb_ref_path = tmp_awb.name
                    awb_ref_pil = to_pil_from_any(awb_ref_image[0])
                    awb_ref_pil.save(awb_ref_path, "PNG")
                    args.awb_ref = awb_ref_path
            else:
                args.awb_ref = None

            # Non-semantic attack
            args.non_semantic = kwargs.get("apply_non_semantic", False)
            args.ns_num_iter = kwargs.get("ns_num_iter", 10)
            args.ns_epsilon = kwargs.get("ns_epsilon", 0.03)

            # LUT
            args.lut = kwargs.get("lut_path", "")
            if not args.lut:
                args.lut = None

            # Blend
            args.blend = kwargs.get("apply_blend", False)
            args.blend_alpha = kwargs.get("blend_alpha", 0.5)
            blend_image = kwargs.get("blend_image")
            if blend_image is not None and args.blend:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_blend:
                    blend_path = tmp_blend.name
                    blend_pil = to_pil_from_any(blend_image[0])
                    blend_pil.save(blend_path, "PNG")
                    args.blend_img = blend_path
            else:
                args.blend_img = None

            # EXIF
            args.fake_exif = kwargs.get("add_fake_exif", False)

            # Run the original processing pipeline
            process_image(input_path, output_path, args)

            # Load the result
            output_img = Image.open(output_path).convert("RGB")
            img_out = np.array(output_img)

            # Convert to tensor [0, 1] range
            img_float = img_out.astype(np.float32) / 255.0
            tensor_out = torch.from_numpy(img_float).to(dtype=torch.float32).unsqueeze(0)
            tensor_out = torch.clamp(tensor_out, 0.0, 1.0)

            # Extract EXIF data if available
            try:
                from PIL.ExifTags import TAGS
                exif_dict = output_img._getexif() or {}
                exif_str = "\n".join([f"{TAGS.get(k, k)}: {v}" for k, v in exif_dict.items()])
            except:
                exif_str = "No EXIF data"

        finally:
            # Cleanup temporary files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            if hasattr(args, 'fft_ref') and args.fft_ref and os.path.exists(args.fft_ref):
                os.unlink(args.fft_ref)
            if hasattr(args, 'awb_ref') and args.awb_ref and os.path.exists(args.awb_ref):
                os.unlink(args.awb_ref)
            if hasattr(args, 'blend_img') and args.blend_img and os.path.exists(args.blend_img):
                os.unlink(args.blend_img)

        return (tensor_out, exif_str)


NODE_CLASS_MAPPINGS = {
    "NovaNodesBatch": NovaNodesBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovaNodesBatch": "Nova Nodes (Batch)"
}
