import logging
import torch


class CLIPTextEncodeBatch:
    """
    Encodes a batch of text prompts using CLIP.
    Uses INPUT_IS_LIST to receive all prompts at once, not one at a time.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"forceInput": True}),  
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_batch"
    INPUT_IS_LIST = True  # Critical: receive ALL prompts at once
    OUTPUT_IS_LIST = (False,)  # Output single batched conditioning

    CATEGORY = "conditioning"

    def encode_batch(self, clip, text):
        """
        Encode list of prompts into batched conditioning.
        
        Args:
            clip: List with single CLIP model [clip_model]
            text: List of prompt strings ["prompt1", "prompt2", ...]
        """
        logger = logging.getLogger("CLIPTextEncodeBatch")
        
        # Extract CLIP model from list (INPUT_IS_LIST wraps everything)
        clip_model = clip[0]
        
        # text is already a list of strings
        prompts = text
        batch_size = len(prompts)
        
        logger.info(f"Encoding batch of {batch_size} prompts")
        logger.debug(f"First prompt: {prompts[0][:80]}...")
        
        # Encode each prompt individually
        cond_tensors = []
        pooled_tensors = []
        
        for idx, prompt_text in enumerate(prompts):
            try:
                logger.debug(f"[{idx+1}/{batch_size}] Encoding prompt...")
                
                tokens = clip_model.tokenize(prompt_text)
                cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
                
                cond_tensors.append(cond)
                pooled_tensors.append(pooled)
                
            except Exception as e:
                logger.error(f"Error encoding prompt {idx + 1}/{batch_size}: {e}", exc_info=True)
                raise
        
        # Find maximum sequence length across all tensors
        max_seq_len = max(cond.shape[1] for cond in cond_tensors)
        logger.debug(f"Max sequence length: {max_seq_len}")

        # Pad all tensors to the same sequence length
        padded_cond_tensors = []
        for idx, cond in enumerate(cond_tensors):
            current_seq_len = cond.shape[1]
            if current_seq_len < max_seq_len:
                # Pad along dimension 1 (sequence dimension)
                # torch.nn.functional.pad format: (left, right, top, bottom, front, back)
                # For 3D tensor [batch, seq, embed]: pad as (0, 0, 0, pad_amount)
                pad_amount = max_seq_len - current_seq_len
                padded = torch.nn.functional.pad(cond, (0, 0, 0, pad_amount), mode='constant', value=0)
                logger.debug(f"Padded tensor {idx} from {current_seq_len} to {max_seq_len}")
                padded_cond_tensors.append(padded)
            else:
                padded_cond_tensors.append(cond)

        # Concatenate all conditioning tensors along batch dimension
        batched_cond = torch.cat(padded_cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"✓ Created batched conditioning: {batched_cond.shape}, pooled: {batched_pooled.shape}")
        
        return ([[batched_cond, {"pooled_output": batched_pooled}]],)


class CLIPTextEncodeSequence:
    """
    Alternative: Encode batch of prompts, output as separate conditionings.
    Use this if you want to process each conditioning separately downstream.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_sequence"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)  # Output list of individual conditionings

    CATEGORY = "conditioning"

    def encode_sequence(self, clip, text):
        """Encode each prompt, return as list of separate conditionings."""
        logger = logging.getLogger("CLIPTextEncodeSequence")
        
        clip_model = clip[0]
        prompts = text
        
        logger.info(f"Encoding {len(prompts)} prompts as sequence")
        
        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            tokens = clip_model.tokenize(prompt_text)
            cond, pooled = clip_model.encode_from_tokens(tokens, return_pooled=True)
            conditionings.append([[cond, {"pooled_output": pooled}]])
        
        logger.info(f"✓ Created {len(conditionings)} separate conditionings")
        return (conditionings,)


class PromptDebugger:
    """
    Debug what data structure is being passed.
    Also uses INPUT_IS_LIST to see everything at once.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "debug"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def debug(self, text):
        """Print detailed information about received data."""
        logger = logging.getLogger("PromptDebugger")
        
        logger.info("=" * 60)
        logger.info("PROMPT DEBUGGER (with INPUT_IS_LIST)")
        logger.info("=" * 60)
        logger.info(f"Received type: {type(text)}")
        logger.info(f"Is list: {isinstance(text, list)}")
        logger.info(f"Length: {len(text)}")
        
        for idx, item in enumerate(text):
            logger.info(f"  [{idx}] {item}")
        
        logger.info("=" * 60)
        
        # Pass through unchanged
        return (text,)


class BatchSizeChecker:
    """
    Verify batch sizes match.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "text", "info")
    FUNCTION = "check"
    INPUT_IS_LIST = (False, True)  # images: single, text: list
    OUTPUT_IS_LIST = (False, True, False)

    CATEGORY = "utils"

    def check(self, images, text):
        """Check batch sizes."""
        logger = logging.getLogger("BatchSizeChecker")
        
        image_count = images.shape[0]
        prompt_count = len(text)
        
        info = f"Images: {image_count}, Prompts: {prompt_count}"
        
        if image_count == prompt_count:
            logger.info(f"✓ Batch sizes MATCH: {info}")
        else:
            logger.warning(f"✗ Batch size MISMATCH: {info}")
        
        return (images, text, info)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBatch": CLIPTextEncodeBatch,
    "CLIPTextEncodeSequence": CLIPTextEncodeSequence,
    "PromptDebugger": PromptDebugger,
    "BatchSizeChecker": BatchSizeChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "CLIPTextEncodeSequence": "CLIP Text Encode (Sequence)",
    "PromptDebugger": "Prompt Debugger",
    "BatchSizeChecker": "Batch Size Checker",
}
