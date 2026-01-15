"""
Batch-compatible version of NovaNodes from Image-Detection-Bypass-Utility
Original: https://github.com/PurinNyova/Image-Detection-Bypass-Utility

This wrapper adds batch processing capability while preserving all original
functionality and exact parameter names/defaults.
"""

import torch
import numpy as np
from PIL import Image
import tempfile
import os
import logging
import json

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


class CameraOptionsNodeBatch:
    """Camera simulation options - batch compatible version."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_bayer": ("BOOLEAN", {"default": False}),
                "apply_jpeg_cycles_o": ("BOOLEAN", {"default": True}),
                "jpeg_cycles": ("INT", {"default": 6, "min": 1, "max": 10}),
                "jpeg_quality": ("INT", {"default": 70, "min": 10, "max": 100}),
                "jpeg_qmax": ("INT", {"default": 95, "min": 10, "max": 100}),
                "apply_vignette_o": ("BOOLEAN", {"default": True}),
                "vignette_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_chromatic_aberration_o": ("BOOLEAN", {"default": True}),
                "ca_shift": ("FLOAT", {"default": 1.20, "min": 0.0, "max": 5.0, "step": 0.01}),
                "iso_scale": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 16.0, "step": 0.01}),
                "read_noise": ("FLOAT", {"default": 2.00, "min": 0.0, "max": 50.0, "step": 0.1}),
                "hot_pixel_prob": ("FLOAT", {"default": 0.0000001, "min": 0.0, "max": 0.001, "step": 0.0000001}),
                "apply_banding_o": ("BOOLEAN", {"default": True}),
                "banding_strength": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_motion_blur_o": ("BOOLEAN", {"default": False}),
                "motion_blur_ksize": ("INT", {"default": 1, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("CAMERAOPT",)
    FUNCTION = "get_cam_opts"
    CATEGORY = "image/postprocessing"

    def get_cam_opts(self, **kwargs):
        """Returns JSON string with camera options."""
        return (json.dumps(kwargs),)


class NSOptionsNodeBatch:
    """Non-semantic attack options - batch compatible version."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "non_semantic": ("BOOLEAN", {"default": False}),
                "ns_iterations": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "ns_learning_rate": ("FLOAT", {"default": 0.0003, "min": 0.000001, "max": 1.0, "step": 0.0001}),
                "ns_t_lpips": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ns_t_l2": ("FLOAT", {"default": 0.00003, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "ns_c_lpips": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ns_c_l2": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 10.0, "step": 0.01}),
                "ns_grad_clip": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("NONSEMANTICOP",)
    FUNCTION = "get_ns_opts"
    CATEGORY = "image/postprocessing"

    def get_ns_opts(self, **kwargs):
        """Returns JSON string with non-semantic options."""
        return (json.dumps(kwargs),)


class NovaNodesBatch:
    """Main image postprocessing node - batch compatible version."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "Cam_Opt": ("CAMERAOPT",),
                "NS_Opt": ("NONSEMANTICOP",),
                "apply_noise_o": ("BOOLEAN", {"default": True}),
                "noise_std_frac": ("FLOAT", {"default": 0.020, "min": 0.0, "max": 0.1, "step": 0.001}),
                "apply_clahe_o": ("BOOLEAN", {"default": True}),
                "clahe_clip": ("FLOAT", {"default": 2.00, "min": 0.5, "max": 10.0, "step": 0.1}),
                "clahe_grid": ("INT", {"default": 8, "min": 2, "max": 32}),
                "fourier_cutoff": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_fourier_o": ("BOOLEAN", {"default": True}),
                "fourier_strength": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fourier_randomness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_phase_perturb": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fourier_radial_smooth": ("INT", {"default": 0, "min": 0, "max": 50}),
                "fourier_mode": (["auto", "ref", "model"], {"default": "auto"}),
                "fourier_alpha": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 4.0, "step": 0.01}),
                "perturb_mag_frac": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.05, "step": 0.001}),
                "enable_awb": ("BOOLEAN", {"default": True}),
                "enable_lut": ("BOOLEAN", {"default": False}),
                "lut": ("STRING", {"default": "X://insert/path/here(.png/.npy/.cube)"}),
                "lut_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glcm": ("BOOLEAN", {"default": False}),
                "glcm_distances": ("STRING", {"default": "1,3"}),
                "glcm_angles": ("STRING", {"default": "0,0.785398,1.5708,2.35619"}),
                "glcm_levels": ("INT", {"default": 256, "min": 2, "max": 65536}),
                "glcm_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lbp": ("BOOLEAN", {"default": False}),
                "lbp_radius": ("INT", {"default": 1, "min": 1, "max": 50}),
                "lbp_n_points": ("INT", {"default": 8, "min": 1, "max": 512}),
                "lbp_method": (["default", "ror", "uniform", "var"], {"default": "uniform"}),
                "lbp_strength": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "apply_exif_o": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "awb_ref_image": ("IMAGE",),
                "fft_ref_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "EXIF")
    FUNCTION = "process_batch"
    CATEGORY = "image/postprocessing"

    def process_batch(self, image, Cam_Opt, NS_Opt, **kwargs):
        """Process entire batch of images through NovaNodes pipeline."""
        logger = logging.getLogger("NovaNodesBatch")

        if not NOVA_AVAILABLE:
            raise ImportError("image_postprocess module not found. Install Image-Detection-Bypass-Utility first.")

        # Parse option JSON strings
        cam_opts = json.loads(Cam_Opt)
        ns_opts = json.loads(NS_Opt)

        # Get batch size
        batch_size = image.shape[0]
        logger.info(f"Processing batch of {batch_size} images through NovaNodes")

        # Process each image in the batch
        processed_tensors = []
        exif_data_list = []

        for idx in range(batch_size):
            logger.debug(f"Processing image {idx + 1}/{batch_size}")

            # Extract single image from batch
            single_image = image[idx:idx+1]

            # Process this image
            processed_tensor, exif_data = self._process_single(
                single_image, idx, cam_opts, ns_opts, **kwargs
            )

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

    def _process_single(self, image, idx, cam_opts, ns_opts, **kwargs):
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

            # Camera options (from JSON)
            args.sim_camera = cam_opts.get("enable_bayer") or cam_opts.get("apply_jpeg_cycles_o") or \
                             cam_opts.get("apply_vignette_o") or cam_opts.get("apply_chromatic_aberration_o") or \
                             cam_opts.get("apply_banding_o") or cam_opts.get("apply_motion_blur_o")
            args.sim_bayer_pattern = cam_opts.get("enable_bayer", False)
            args.sim_jpeg_cycles = cam_opts.get("jpeg_cycles", 1) if cam_opts.get("apply_jpeg_cycles_o", False) else 0
            args.sim_jpeg_quality = cam_opts.get("jpeg_quality", 70)
            args.sim_jpeg_qmax = cam_opts.get("jpeg_qmax", 95)
            args.sim_vignetting = cam_opts.get("apply_vignette_o", False)
            args.sim_vignette_strength = cam_opts.get("vignette_strength", 0.35)
            args.sim_chrom_aberration = cam_opts.get("apply_chromatic_aberration_o", False)
            args.sim_ca_shift = cam_opts.get("ca_shift", 1.20)
            args.sim_iso_scale = cam_opts.get("iso_scale", 1.0)
            args.sim_read_noise = cam_opts.get("read_noise", 2.0)
            args.sim_hot_pixel_prob = cam_opts.get("hot_pixel_prob", 1e-7)
            args.sim_banding = cam_opts.get("apply_banding_o", False)
            args.sim_banding_strength = cam_opts.get("banding_strength", 0.0)
            args.sim_motion_blur = cam_opts.get("apply_motion_blur_o", False)
            args.sim_motion_blur_ksize = cam_opts.get("motion_blur_ksize", 1)

            # Non-semantic options (from JSON)
            args.non_semantic = ns_opts.get("non_semantic", False)
            args.ns_num_iter = ns_opts.get("ns_iterations", 10)
            args.ns_learning_rate = ns_opts.get("ns_learning_rate", 0.0003)
            args.ns_t_lpips = ns_opts.get("ns_t_lpips", 0.04)
            args.ns_t_l2 = ns_opts.get("ns_t_l2", 0.00003)
            args.ns_c_lpips = ns_opts.get("ns_c_lpips", 0.01)
            args.ns_c_l2 = ns_opts.get("ns_c_l2", 0.6)
            args.ns_grad_clip = ns_opts.get("ns_grad_clip", 0.05)

            # Main parameters (from kwargs)
            args.noise = kwargs.get("apply_noise_o", True)
            args.noise_std_frac = kwargs.get("noise_std_frac", 0.02)

            args.clahe = kwargs.get("apply_clahe_o", True)
            args.clahe_clip_limit = kwargs.get("clahe_clip", 2.0)
            args.clahe_tile_size = kwargs.get("clahe_grid", 8)

            args.fft = kwargs.get("apply_fourier_o", True)
            args.fft_cutoff = kwargs.get("fourier_cutoff", 0.25)
            args.fft_strength = kwargs.get("fourier_strength", 0.9)
            args.fft_randomness = kwargs.get("fourier_randomness", 0.05)
            args.fft_phase_perturb = kwargs.get("fourier_phase_perturb", 0.04)
            args.fft_radial_smooth = kwargs.get("fourier_radial_smooth", 0)
            args.fft_mode = kwargs.get("fourier_mode", "auto")
            args.fft_alpha = kwargs.get("fourier_alpha", 1.0)

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

            args.perturb = kwargs.get("perturb_mag_frac", 0.001) > 0.0
            args.perturb_max_shift = int(kwargs.get("perturb_mag_frac", 0.001) * 100)

            args.awb = kwargs.get("enable_awb", True)
            awb_ref_image = kwargs.get("awb_ref_image")
            if awb_ref_image is not None and args.awb:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_awb:
                    awb_ref_path = tmp_awb.name
                    awb_ref_pil = to_pil_from_any(awb_ref_image[0])
                    awb_ref_pil.save(awb_ref_path, "PNG")
                    args.awb_ref = awb_ref_path
            else:
                args.awb_ref = None

            lut_path = kwargs.get("lut", "")
            args.lut = lut_path if kwargs.get("enable_lut", False) and lut_path and lut_path != "X://insert/path/here(.png/.npy/.cube)" else None
            args.lut_strength = kwargs.get("lut_strength", 1.0)

            args.glcm = kwargs.get("glcm", False)
            args.glcm_distances = [int(x.strip()) for x in kwargs.get("glcm_distances", "1,3").split(",")]
            args.glcm_angles = [float(x.strip()) for x in kwargs.get("glcm_angles", "0,0.785398,1.5708,2.35619").split(",")]
            args.glcm_levels = kwargs.get("glcm_levels", 256)
            args.glcm_strength = kwargs.get("glcm_strength", 0.5)

            args.lbp = kwargs.get("lbp", False)
            args.lbp_radius = kwargs.get("lbp_radius", 1)
            args.lbp_n_points = kwargs.get("lbp_n_points", 8)
            args.lbp_method = kwargs.get("lbp_method", "uniform")
            args.lbp_strength = kwargs.get("lbp_strength", 0.3)

            args.seed = kwargs.get("seed", -1)
            args.fake_exif = kwargs.get("apply_exif_o", True)

            # Blend option (not in UI but needed for compatibility)
            args.blend = False
            args.blend_img = None

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

        return (tensor_out, exif_str)


NODE_CLASS_MAPPINGS = {
    "CameraOptionsNodeBatch": CameraOptionsNodeBatch,
    "NSOptionsNodeBatch": NSOptionsNodeBatch,
    "NovaNodesBatch": NovaNodesBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraOptionsNodeBatch": "Camera Options (NOVA)",
    "NSOptionsNodeBatch": "Non-semantic Options (NOVA)",
    "NovaNodesBatch": "Image Postprocess (NOVA NODES)",
}
