"""
Batch-compatible ZImageFunControlnet node.

The original ZImageFunControlnet (from comfy_extras/nodes_model_patch.py) uses a
ZImageControlPatch that encodes a single conditioning image and applies it to all
batch items uniformly.

The problem with batches:
  - You have N images, each needing its OWN ControlNet reference (e.g. N different
    HED maps, one per source image).
  - The ZImageControlPatch stores encoded_image with shape [N, ...].
  - ComfyUI's CFG sampler doubles the internal batch to 2N (cond + uncond), so the
    control model receives txt/pe/vec of shape [2N, ...] but encoded_image is [N, ...].
  - This shape mismatch causes a crash.

The fix (ZImageBatchControlPatch):
  - Before calling the control model, detect the CFG doubling.
  - If txt.shape[0] is an exact multiple of encoded_image.shape[0], repeat the
    conditioning to match (e.g. [N, ...] -> [2N, ...]).
  - Everything else stays identical to the original ZImageControlPatch logic.
"""

import torch
import comfy.ldm.lumina.controlnet
import comfy.utils
import comfy.model_management

from comfy_extras.nodes_model_patch import ZImageControlPatch, DiffSynthCnetPatch


class ZImageBatchControlPatch(ZImageControlPatch):
    """
    Extends ZImageControlPatch to handle batched image inputs.

    Step-by-step what happens during each denoising forward pass:

    1.  __call__ receives kwargs from the sampler:
          x          – current noisy latent   [B, C, H, W]
          img        – image token features   [B, T, D]
          img_input  – original image tokens  [B, T, D]
          txt        – text conditioning      [B, T_txt, D]  (B = 2N when CFG)
          pe         – positional encoding    [B, ...]
          vec        – pooled conditioning    [B, D]
          block_index – which transformer block we are in

    2.  If the spatial size changed, re-encode the control image(s) with the VAE.

    3.  --- BATCH FIX ---
        Check if txt.shape[0] > encoded_image.shape[0].
        If yes and it is an exact multiple (e.g. 2N vs N), call
        repeat_interleave so encoded becomes [2N, ...] matching txt.

    4.  On the first block of each denoising step (temp_data is None / reset),
        run the full ZImage_Control forward to produce all block hidden states.

    5.  Advance the control block counter and apply the control residual to img.

    6.  Return the modified kwargs.
    """

    def __call__(self, kwargs):
        x          = kwargs.get("x")
        img        = kwargs.get("img")
        img_input  = kwargs.get("img_input")
        txt        = kwargs.get("txt")
        pe         = kwargs.get("pe")
        vec        = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        block_type  = kwargs.get("block_type", "")

        spacial_compression = self.vae.spacial_compression_encode()

        # --- Re-encode if spatial resolution changed ---
        if self.encoded_image is None or self.encoded_image_size != (
            x.shape[-2] * spacial_compression,
            x.shape[-1] * spacial_compression,
        ):
            image_scaled = None
            if self.image is not None:
                image_scaled = comfy.utils.common_upscale(
                    self.image.movedim(-1, 1),
                    x.shape[-1] * spacial_compression,
                    x.shape[-2] * spacial_compression,
                    "area", "center",
                ).movedim(1, -1)
                self.encoded_image_size = (image_scaled.shape[-3], image_scaled.shape[-2])

            inpaint_scaled = None
            if self.inpaint_image is not None:
                inpaint_scaled = comfy.utils.common_upscale(
                    self.inpaint_image.movedim(-1, 1),
                    x.shape[-1] * spacial_compression,
                    x.shape[-2] * spacial_compression,
                    "area", "center",
                ).movedim(1, -1)
                self.encoded_image_size = (inpaint_scaled.shape[-3], inpaint_scaled.shape[-2])

            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled, inpaint_scaled)
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_blocks     = self.model_patch.model.n_control_layers
        div             = round(30 / cnet_blocks)
        cnet_index      = block_index // div
        cnet_index_float = block_index / div

        kwargs.pop("img")   # modified in-place below
        kwargs.pop("txt")

        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        # --- BATCH FIX: expand encoded_image to match the CFG-doubled batch ---
        # Example: 4 source images -> encoded_image [4, T, D]
        #          CFG sampler passes txt [8, T_txt, D]  (4 cond + 4 uncond)
        # We repeat_interleave to get encoded [8, T, D] so the control model
        # sees matching batch sizes.
        encoded = self.encoded_image.to(img.dtype)
        if txt.shape[0] != encoded.shape[0] and encoded.shape[0] > 0:
            if txt.shape[0] % encoded.shape[0] == 0:
                repeat_factor = txt.shape[0] // encoded.shape[0]
                encoded = encoded.repeat_interleave(repeat_factor, dim=0)

        # --- Run control model forward (once per denoising step) ---
        if self.temp_data is None or self.temp_data[0] > cnet_index:
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt, encoded, pe, vec)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt, encoded, pe, vec)))

        # --- Apply control residual to img ---
        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (
                next_layer,
                self.model_patch.model.forward_noise_refiner_block(
                    block_index,
                    self.temp_data[1][1],
                    img_input[:, :self.temp_data[1][1].shape[1]],
                    None, pe, vec,
                ),
            )
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += self.temp_data[1][0] * self.strength
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (
                    next_layer,
                    self.model_patch.model.forward_control_block(
                        next_layer,
                        self.temp_data[1][1],
                        img_input[:, :self.temp_data[1][1].shape[1]],
                        None, pe, vec,
                    ),
                )

            if cnet_index_float == self.temp_data[0]:
                img[:, :self.temp_data[1][0].shape[1]] += self.temp_data[1][0] * self.strength
                if cnet_blocks == self.temp_data[0] + 1:
                    self.temp_data = None

        return kwargs


class ZImageFunControlnetBatch:
    """
    Batch-compatible ZImageFunControlnet node.

    Inputs
    ------
    model        – the base MODEL to patch
    model_patch  – MODEL_PATCH loaded by ModelPatchLoader
    vae          – VAE used to encode the control image(s) into latent space
    strength     – how strongly the ControlNet influences generation (default 1.0)
    image        – (optional) batch of control images, one per source image
                   e.g. 4 HED maps for a batch of 4 source images [4, H, W, C]
    inpaint_image – (optional) batch of inpaint reference images [N, H, W, C]
    mask         – (optional) batch of inpaint masks [N, H, W] or [N, 1, H, W]

    Output
    ------
    model  – the patched MODEL with per-image ControlNet conditioning applied

    How it works
    ------------
    1. Clones the input model.
    2. Strips any alpha channel from image/inpaint_image.
    3. Normalises mask dimensions (ComfyUI masks need an extra channel dim).
    4. Inverts the mask (ComfyUI convention: 0 = keep, 1 = inpaint ->
       internally we need 1 = keep, 0 = inpaint).
    5. Creates a ZImageBatchControlPatch (which handles CFG batch doubling).
    6. Attaches it as both a noise_refiner patch and a double_block patch on the
       cloned model so ZImage's two-pass control mechanism is respected.
    7. Returns the patched model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":       ("MODEL",),
                "model_patch": ("MODEL_PATCH",),
                "vae":         ("VAE",),
                "strength":    ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "image":         ("IMAGE",),
                "inpaint_image": ("IMAGE",),
                "mask":          ("MASK",),
            },
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "apply_batch_controlnet"
    CATEGORY      = "I9/batch"

    def apply_batch_controlnet(
        self,
        model,
        model_patch,
        vae,
        strength: float = 1.0,
        image=None,
        inpaint_image=None,
        mask=None,
    ):
        model_patched = model.clone()

        # Strip alpha channel – ZImage only uses RGB
        if image is not None:
            image = image[:, :, :, :3]
        if inpaint_image is not None:
            inpaint_image = inpaint_image[:, :, :, :3]

        # Normalise mask shape: ComfyUI delivers [N, H, W] or [N, 1, H, W];
        # ZImageControlPatch expects [N, 1, 1, H, W] and inverted polarity.
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)   # [N, H, W] -> [N, 1, H, W]
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)   # [N, 1, H, W] -> [N, 1, 1, H, W]
            mask = 1.0 - mask              # invert: ComfyUI 0=keep -> internal 1=keep

        if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
            patch = ZImageBatchControlPatch(
                model_patch, vae, image, strength,
                inpaint_image=inpaint_image, mask=mask,
            )
            # ZImage uses both a noise_refiner pass and the main double_block pass
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        else:
            # Fall back to standard DiffSynth patch for non-ZImage model_patches
            model_patched.set_model_double_block_patch(
                DiffSynthCnetPatch(model_patch, vae, image, strength, mask)
            )

        return (model_patched,)


NODE_CLASS_MAPPINGS = {
    "ZImageFunControlnetBatch": ZImageFunControlnetBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageFunControlnetBatch": "ZImage Fun Controlnet (Batch)",
}
