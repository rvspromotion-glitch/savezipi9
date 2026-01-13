import logging
import torch


class CLIPTextEncodeBatch:
    """
    Encodes a batch of text prompts using CLIP.
    Accepts the list output from Ask_Gemini_Batch and creates batched conditioning.
    Works with any batch size.
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
    FUNCTION = "encode_batch"

    CATEGORY = "conditioning"

    def encode_batch(self, clip, prompts):
        """
        Encode list of prompts into batched conditioning.
        This works because OUTPUT_IS_LIST in Gemini node passes Python list directly.
        """
        logger = logging.getLogger("CLIPTextEncodeBatch")
        
        # Debug what we received
        logger.info(f"Received prompts - Type: {type(prompts)}")
        
        # The prompts come as a Python list directly from the Gemini node
        # because it uses OUTPUT_IS_LIST = (True,)
        if not isinstance(prompts, list):
            logger.warning(f"Expected list, got {type(prompts)}, converting to list")
            prompts = [prompts]
        
        batch_size = len(prompts)
        logger.info(f"Encoding batch of {batch_size} prompts")
        
        # Encode each prompt individually
        cond_tensors = []
        pooled_tensors = []
        
        for idx, prompt_text in enumerate(prompts):
            try:
                # Convert to string in case it's not
                prompt_str = str(prompt_text)
                logger.debug(f"[{idx+1}/{batch_size}] Encoding: {prompt_str[:80]}...")
                
                tokens = clip.tokenize(prompt_str)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                
                cond_tensors.append(cond)
                pooled_tensors.append(pooled)
                
            except Exception as e:
                logger.error(f"Error encoding prompt {idx + 1}/{batch_size}: {e}", exc_info=True)
                raise
        
        # Concatenate all conditioning tensors along batch dimension (dim=0)
        batched_cond = torch.cat(cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"✓ Created batched conditioning: {batched_cond.shape}, pooled: {batched_pooled.shape}")
        
        return ([[batched_cond, {"pooled_output": batched_pooled}]],)


class PromptDebugger:
    """
    Debug what data structure is actually being passed between nodes.
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
    FUNCTION = "debug"

    CATEGORY = "utils"

    def debug(self, prompts):
        """Print detailed information about the prompts."""
        logger = logging.getLogger("PromptDebugger")
        
        logger.info("=" * 60)
        logger.info("PROMPT DEBUGGER")
        logger.info("=" * 60)
        logger.info(f"Type: {type(prompts)}")
        logger.info(f"Is list: {isinstance(prompts, list)}")
        
        if isinstance(prompts, list):
            logger.info(f"Length: {len(prompts)}")
            for idx, item in enumerate(prompts):
                logger.info(f"  [{idx}] Type: {type(item)}, Value: {str(item)[:100]}")
        else:
            logger.info(f"Value: {str(prompts)[:200]}")
        
        logger.info("=" * 60)
        
        # Pass through unchanged
        return (prompts,)


class BatchSizeChecker:
    """
    Verify batch sizes match between images and prompts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompts": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "prompts", "info")
    FUNCTION = "check"

    CATEGORY = "utils"

    def check(self, images, prompts):
        """Check and report batch sizes."""
        logger = logging.getLogger("BatchSizeChecker")
        
        image_count = images.shape[0] if hasattr(images, 'shape') else len(images)
        
        if isinstance(prompts, list):
            prompt_count = len(prompts)
        else:
            prompt_count = 1
        
        info = f"Images: {image_count}, Prompts: {prompt_count}"
        
        if image_count == prompt_count:
            logger.info(f"✓ Batch sizes match: {info}")
        else:
            logger.warning(f"✗ Batch size MISMATCH: {info}")
        
        return (images, prompts, info)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBatch": CLIPTextEncodeBatch,
    "PromptDebugger": PromptDebugger,
    "BatchSizeChecker": BatchSizeChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "PromptDebugger": "Prompt Debugger",
    "BatchSizeChecker": "Batch Size Checker",
}
