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
    INPUT_IS_LIST = (False, True)  # CLIP is single, prompts is list
    OUTPUT_IS_LIST = (False,)  # Output single batched conditioning

    CATEGORY = "conditioning"

    def encode_batch(self, clip, prompts):
        """
        Encode multiple prompts into a batched conditioning tensor.
        
        Args:
            clip: Single CLIP model
            prompts: List of prompt strings
            
        Returns:
            Batched CONDITIONING tensor
        """
        logger = logging.getLogger("CLIPTextEncodeBatch")
        
        # Flatten prompts if nested list
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], list):
            prompts = prompts[0]
        
        batch_size = len(prompts)
        logger.info(f"Encoding batch of {batch_size} prompts with CLIP")
        logger.debug(f"First prompt: {prompts[0][:100]}...")
        
        # Encode each prompt
        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            try:
                tokens = clip.tokenize(prompt_text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                conditionings.append([[cond, {"pooled_output": pooled}]])
                logger.debug(f"Encoded prompt {idx + 1}/{batch_size}")
            except Exception as e:
                logger.error(f"Error encoding prompt {idx + 1}: {e}", exc_info=True)
                raise
        
        # Batch the conditioning tensors
        # Stack all cond tensors along batch dimension
        cond_tensors = [c[0][0][0] for c in conditionings]
        pooled_tensors = [c[0][0][1]["pooled_output"] for c in conditionings]
        
        batched_cond = torch.cat(cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"Created batched conditioning: {batched_cond.shape}, pooled: {batched_pooled.shape}")
        
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
    INPUT_IS_LIST = (False, True)
    OUTPUT_IS_LIST = (True,)  # Output list of individual conditionings

    CATEGORY = "conditioning"

    def encode_sequence(self, clip, prompts):
        """
        Encode multiple prompts into individual conditioning tensors.
        Returns a list where each element can be used separately.
        """
        logger = logging.getLogger("CLIPTextEncodeSequence")
        
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], list):
            prompts = prompts[0]
        
        batch_size = len(prompts)
        logger.info(f"Encoding sequence of {batch_size} prompts")
        
        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            tokens = clip.tokenize(prompt_text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditionings.append([[cond, {"pooled_output": pooled}]])
            logger.debug(f"Encoded prompt {idx + 1}/{batch_size}")
        
        return (conditionings,)


class PromptDebugger:
    """
    Debug node to inspect what prompts are being received.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "debug_prompts"
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def debug_prompts(self, prompts):
        """Print prompt information for debugging."""
        logger = logging.getLogger("PromptDebugger")
        
        logger.info(f"Prompts type: {type(prompts)}")
        logger.info(f"Prompts length: {len(prompts) if isinstance(prompts, list) else 'N/A'}")
        
        if isinstance(prompts, list):
            for idx, p in enumerate(prompts):
                logger.info(f"Prompt {idx}: type={type(p)}, value={str(p)[:100]}")
        
        return (prompts,)


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
        
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], list):
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
    "PromptDebugger": PromptDebugger,
    "BatchImageLatentSync": BatchImageLatentSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "CLIPTextEncodeSequence": "CLIP Text Encode (Sequence)",
    "PromptDebugger": "Prompt Debugger",
    "BatchImageLatentSync": "Batch Image/Latent Sync Check",
}
