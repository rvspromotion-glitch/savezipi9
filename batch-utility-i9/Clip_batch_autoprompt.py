import logging
import torch


class CLIPTextEncodeBatch:
    """
    Encodes a batch of text prompts using CLIP.
    Takes a list of prompts and outputs a batch of conditioning tensors.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompts": ("STRING", {"forceInput": True}),  # List of prompts
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_batch"
    INPUT_IS_LIST = True  # Critical: Accept list inputs
    OUTPUT_IS_LIST = (False,)  # Output single batched conditioning

    CATEGORY = "conditioning"

    def encode_batch(self, clip, prompts):
        """
        Encode multiple prompts into a batched conditioning tensor.
        
        Args:
            clip: List containing single CLIP model (due to INPUT_IS_LIST)
            prompts: List of prompt strings
            
        Returns:
            Batched CONDITIONING tensor
        """
        logger = logging.getLogger("CLIPTextEncodeBatch")
        
        # Extract single CLIP model from list
        clip_model = clip[0]
        
        # Flatten prompts if nested
        if isinstance(prompts[0], list):
            prompts = prompts[0]
        
        batch_size = len(prompts)
        logger.info(f"Encoding batch of {batch_size} prompts")
        
        # Encode each prompt
        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            tokens = clip_model.tokenize(prompt_text)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            conditionings.append([[cond, {"pooled_output": pooled}]])
            logger.debug(f"Encoded prompt {idx + 1}/{batch_size}: {prompt_text[:50]}...")
        
        # Batch the conditioning tensors
        # Stack all cond tensors along batch dimension
        cond_tensors = [c[0][0][0] for c in conditionings]
        pooled_tensors = [c[0][0][1]["pooled_output"] for c in conditionings]
        
        batched_cond = torch.cat(cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"Created batched conditioning: {batched_cond.shape}")
        
        return ([[batched_cond, {"pooled_output": batched_pooled}]],)


class CLIPTextEncodeSequence:
    """
    Alternative node that outputs a sequence of individual conditioning tensors.
    Useful if you need to process prompts one at a time in the pipeline.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompts": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_sequence"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)  # Output list of individual conditionings

    CATEGORY = "conditioning"

    def encode_sequence(self, clip, prompts):
        """
        Encode multiple prompts into individual conditioning tensors.
        Returns a list where each element can be used separately.
        """
        logger = logging.getLogger("CLIPTextEncodeSequence")
        
        clip_model = clip[0]
        
        if isinstance(prompts[0], list):
            prompts = prompts[0]
        
        batch_size = len(prompts)
        logger.info(f"Encoding sequence of {batch_size} prompts")
        
        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            tokens = clip_model.tokenize(prompt_text)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            conditionings.append([[cond, {"pooled_output": pooled}]])
            logger.debug(f"Encoded prompt {idx + 1}/{batch_size}")
        
        return (conditionings,)


class BatchImageLatentSync:
    """
    Utility node to verify batch sizes match between images and prompts.
    Helps debug batch alignment issues.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompts": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "prompts", "image_count", "prompt_count")
    FUNCTION = "sync_check"
    INPUT_IS_LIST = (False, True)
    OUTPUT_IS_LIST = (False, True, False, False)

    CATEGORY = "utils"

    def sync_check(self, images, prompts):
        """Check and report batch sizes."""
        logger = logging.getLogger("BatchImageLatentSync")
        
        if isinstance(prompts[0], list):
            prompts = prompts[0]
        
        image_count = images.shape[0]
        prompt_count = len(prompts)
        
        if image_count != prompt_count:
            logger.warning(f"Batch size mismatch! Images: {image_count}, Prompts: {prompt_count}")
        else:
            logger.info(f"Batch sizes match: {image_count} images and {prompt_count} prompts")
        
        return (images, prompts, image_count, prompt_count)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBatch": CLIPTextEncodeBatch,
    "CLIPTextEncodeSequence": CLIPTextEncodeSequence,
    "BatchImageLatentSync": BatchImageLatentSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "CLIPTextEncodeSequence": "CLIP Text Encode (Sequence)",
    "BatchImageLatentSync": "Batch Image/Latent Sync Check",
}
