import logging
import torch
import math


def lcm(a, b):
    """Calculate least common multiple of two numbers."""
    return a * b // math.gcd(a, b)


def lcm_for_list(numbers):
    """Calculate LCM for a list of numbers."""
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = lcm(current_lcm, number)
    return current_lcm


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
                "prompts": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_batch"
    INPUT_IS_LIST = True  # Critical: receive ALL prompts at once
    OUTPUT_IS_LIST = (False,)  # Output single batched conditioning

    CATEGORY = "conditioning"

    def encode_batch(self, clip, prompts):
        """
        Encode list of prompts into batched conditioning.

        Args:
            clip: List with single CLIP model [clip_model]
            prompts: List of prompt strings ["prompt1", "prompt2", ...]
        """
        logger = logging.getLogger("CLIPTextEncodeBatch")

        # Extract CLIP model from list (INPUT_IS_LIST wraps everything)
        clip_model = clip[0]

        # Debug the CLIP object
        logger.info(f"CLIP object type: {type(clip_model)}")
        logger.info(f"CLIP object methods: {[m for m in dir(clip_model) if not m.startswith('_')]}")

        # prompts is already a list of strings
        batch_size = len(prompts)
        
        logger.info(f"Encoding batch of {batch_size} prompts")
        logger.debug(f"First prompt: {prompts[0][:80]}...")
        
        # Encode each prompt individually
        cond_tensors = []
        pooled_tensors = []
        
        for idx, prompt_text in enumerate(prompts):
            try:
                logger.debug(f"[{idx+1}/{batch_size}] Encoding prompt (length: {len(prompt_text)} chars)...")

                # Validate prompt
                if not prompt_text or not prompt_text.strip():
                    logger.warning(f"Prompt {idx+1} is empty or whitespace only. Using fallback.")
                    prompt_text = "empty prompt"

                cond = None
                pooled = None
                method_1_failed = False

                # Method 1: Try direct encode method first
                if hasattr(clip_model, 'encode') and callable(getattr(clip_model, 'encode')):
                    logger.debug(f"[{idx+1}/{batch_size}] Using clip.encode() method")
                    try:
                        result = clip_model.encode(prompt_text)
                        logger.debug(f"[{idx+1}/{batch_size}] encode() returned: {type(result)}")

                        # Handle different return types
                        if torch.is_tensor(result):
                            # encode() returns conditioning tensor directly
                            logger.debug(f"[{idx+1}/{batch_size}] encode() returned raw tensor, shape: {result.shape}")
                            cond = result
                            # For models without pooled output, create zero tensor or None
                            # We'll try to get pooled from encode_from_tokens as fallback
                            pooled = None
                            method_1_failed = True  # Signal we need to get pooled separately

                        elif isinstance(result, list) and len(result) > 0:
                            # encode() returns full conditioning format [[cond, {"pooled_output": pooled}]]
                            if isinstance(result[0], list) and len(result[0]) == 2:
                                cond = result[0][0]
                                pooled = result[0][1].get("pooled_output") if isinstance(result[0][1], dict) else None
                                logger.debug(f"[{idx+1}/{batch_size}] Extracted from list format: cond shape={cond.shape}, pooled={type(pooled)}")
                            else:
                                raise ValueError(f"Unexpected encode() return structure: {result}")
                        else:
                            raise ValueError(f"Unexpected encode() return type: {type(result)}")

                    except Exception as e:
                        logger.warning(f"[{idx+1}/{batch_size}] clip.encode() failed: {e}, trying encode_from_tokens")
                        method_1_failed = True
                        cond = None
                        pooled = None
                else:
                    method_1_failed = True

                # Method 2: Use tokenize + encode_from_tokens (either as primary or to get pooled)
                if method_1_failed or pooled is None:
                    logger.debug(f"[{idx+1}/{batch_size}] Using tokenize + encode_from_tokens method")
                    tokens = clip_model.tokenize(prompt_text)
                    logger.debug(f"[{idx+1}/{batch_size}] Tokenized successfully, tokens type: {type(tokens)}")

                    # Try with return_pooled=True first
                    try:
                        result = clip_model.encode_from_tokens(tokens, return_pooled=True)
                        logger.debug(f"[{idx+1}/{batch_size}] encode_from_tokens(return_pooled=True) returned: type={type(result)}")

                        if isinstance(result, tuple) and len(result) == 2:
                            result_cond, result_pooled = result
                            logger.debug(f"[{idx+1}/{batch_size}] Unpacked: cond type={type(result_cond)}, pooled type={type(result_pooled)}")

                            # Use these if we don't have them from Method 1
                            if cond is None:
                                cond = result_cond
                            if pooled is None and result_pooled is not None:
                                pooled = result_pooled
                        else:
                            logger.warning(f"[{idx+1}/{batch_size}] Unexpected return format, trying without return_pooled")
                            raise ValueError("Force fallback")

                    except Exception as e:
                        logger.debug(f"[{idx+1}/{batch_size}] return_pooled=True failed: {e}, trying without")
                        # Fallback: try without return_pooled
                        result = clip_model.encode_from_tokens(tokens, return_pooled=False)
                        logger.debug(f"[{idx+1}/{batch_size}] encode_from_tokens(return_pooled=False) returned: type={type(result)}")

                        if cond is None:
                            if torch.is_tensor(result):
                                cond = result
                            elif isinstance(result, tuple):
                                cond = result[0]
                            else:
                                cond = result

                # Final validation and fallback for pooled
                if cond is None:
                    raise ValueError(f"Failed to get conditioning tensor for prompt {idx+1}")

                if pooled is None:
                    # Create zero pooled tensor as fallback
                    logger.warning(f"[{idx+1}/{batch_size}] No pooled output available, creating zero tensor")
                    # Typically pooled is [batch_size, hidden_dim], use same hidden dim as cond
                    hidden_dim = cond.shape[-1] if len(cond.shape) >= 2 else 768
                    pooled = torch.zeros((cond.shape[0], hidden_dim), dtype=cond.dtype, device=cond.device)
                    logger.debug(f"[{idx+1}/{batch_size}] Created zero pooled tensor: shape={pooled.shape}")

                logger.debug(f"[{idx+1}/{batch_size}] Encoded: cond shape={cond.shape}, pooled shape={pooled.shape}")

                cond_tensors.append(cond)
                pooled_tensors.append(pooled)

            except Exception as e:
                logger.error(f"Error encoding prompt {idx + 1}/{batch_size}: {e}", exc_info=True)
                logger.error(f"Problematic prompt: {repr(prompt_text)}")
                raise
        
        # Validate we have tensors before processing
        if not cond_tensors or not pooled_tensors:
            raise RuntimeError("No valid conditioning tensors were created")

        if len(cond_tensors) != len(pooled_tensors):
            raise RuntimeError(f"Mismatch: {len(cond_tensors)} cond tensors but {len(pooled_tensors)} pooled tensors")

        # Verify no None values slipped through
        for idx, (cond, pooled) in enumerate(zip(cond_tensors, pooled_tensors)):
            if cond is None or pooled is None:
                raise RuntimeError(f"None tensor found at index {idx}: cond={cond}, pooled={pooled}")

        # Return as list of separate conditioning items for proper batch pairing
        # RES4LYF/ClownKSampler expects: [[cond0, {}], [cond1, {}], [cond2, {}]]
        # Each conditioning keeps its natural length (no padding) to match non-batch behavior
        conditionings = []
        for idx, (cond, pooled) in enumerate(zip(cond_tensors, pooled_tensors)):
            conditionings.append([cond, {"pooled_output": pooled}])
            logger.debug(f"Conditioning {idx}: shape {cond.shape}, pooled shape {pooled.shape}")

        logger.info(f"✓ Created {len(conditionings)} separate conditionings for batch pairing")

        # Return as list (not wrapped in extra brackets)
        return (conditionings,)


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
        """Encode each prompt, return as list of separate conditionings."""
        logger = logging.getLogger("CLIPTextEncodeSequence")

        clip_model = clip[0]
        batch_size = len(prompts)

        logger.info(f"Encoding {batch_size} prompts as sequence")

        conditionings = []
        for idx, prompt_text in enumerate(prompts):
            try:
                logger.debug(f"[{idx+1}/{batch_size}] Encoding prompt...")

                # Use the same robust encoding logic as CLIPTextEncodeBatch
                if not prompt_text or not prompt_text.strip():
                    logger.warning(f"Prompt {idx+1} is empty, using fallback")
                    prompt_text = "empty prompt"

                cond = None
                pooled = None

                # Try direct encode first
                if hasattr(clip_model, 'encode'):
                    try:
                        result = clip_model.encode(prompt_text)
                        if torch.is_tensor(result):
                            cond = result
                    except:
                        pass

                # Fallback to tokenize method
                if cond is None:
                    tokens = clip_model.tokenize(prompt_text)
                    result = clip_model.encode_from_tokens(tokens, return_pooled=True)

                    if isinstance(result, tuple) and len(result) == 2:
                        cond, pooled = result
                    else:
                        cond = result if torch.is_tensor(result) else result[0]

                # Create zero pooled if None
                if pooled is None:
                    logger.debug(f"[{idx+1}/{batch_size}] Creating zero pooled tensor")
                    hidden_dim = cond.shape[-1] if len(cond.shape) >= 2 else 768
                    pooled = torch.zeros((cond.shape[0], hidden_dim), dtype=cond.dtype, device=cond.device)

                logger.debug(f"[{idx+1}/{batch_size}] Encoded: cond shape={cond.shape}, pooled shape={pooled.shape}")
                conditionings.append([[cond, {"pooled_output": pooled}]])

            except Exception as e:
                logger.error(f"Error encoding prompt {idx+1}/{batch_size}: {e}", exc_info=True)
                raise

        logger.info(f"✓ Created {len(conditionings)} separate conditionings")
        return (conditionings,)


class ConditioningSelector:
    """
    Selects a single conditioning from a list based on index.
    Use with I9 Batch Processing in Sequential mode to pair images with prompts 1-to-1.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "select"
    INPUT_IS_LIST = (True, False)  # conditioning is list, index is single value
    OUTPUT_IS_LIST = (False,)  # Output single conditioning

    CATEGORY = "conditioning"

    def select(self, conditioning, index):
        """Select conditioning at specified index."""
        logger = logging.getLogger("ConditioningSelector")

        cond_list = conditioning

        # Handle index being either int or list[int] (depends on upstream node)
        if isinstance(index, list):
            if len(index) == 0:
                raise ValueError("Index list is empty")
            idx = index[0]
            logger.debug(f"Extracted index {idx} from list {index}")
        else:
            idx = index

        logger.info(f"Selecting conditioning at index {idx} from list of {len(cond_list)} conditionings")

        if idx < 0 or idx >= len(cond_list):
            logger.error(f"Index {idx} out of range [0, {len(cond_list)-1}]")
            raise IndexError(f"Conditioning index {idx} out of range, only {len(cond_list)} conditionings available")

        selected = cond_list[idx]
        logger.info(f"✓ Selected conditioning {idx}")

        return (selected,)


class PromptDebugger:
    """
    Debug what data structure is being passed.
    Also uses INPUT_IS_LIST to see everything at once.
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
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def debug(self, prompts):
        """Print detailed information about received data."""
        logger = logging.getLogger("PromptDebugger")

        logger.info("=" * 60)
        logger.info("PROMPT DEBUGGER (with INPUT_IS_LIST)")
        logger.info("=" * 60)
        logger.info(f"Received type: {type(prompts)}")
        logger.info(f"Is list: {isinstance(prompts, list)}")
        logger.info(f"Count: {len(prompts)}")
        logger.info("")

        for idx, item in enumerate(prompts):
            item_str = str(item)
            logger.info(f"Prompt [{idx}]:")
            logger.info(f"  Length: {len(item_str)} characters")
            logger.info(f"  Preview: {item_str[:100]}{'...' if len(item_str) > 100 else ''}")
            logger.info(f"  Full text: {item_str}")
            logger.info("")

        logger.info("=" * 60)

        # Pass through unchanged
        return (prompts,)


class BatchSizeChecker:
    """
    Verify batch sizes match.
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
    INPUT_IS_LIST = (False, True)  # images: single, prompts: list
    OUTPUT_IS_LIST = (False, True, False)

    CATEGORY = "utils"

    def check(self, images, prompts):
        """Check batch sizes."""
        logger = logging.getLogger("BatchSizeChecker")

        image_count = images.shape[0]
        prompt_count = len(prompts)

        info = f"Images: {image_count}, Prompts: {prompt_count}"

        if image_count == prompt_count:
            logger.info(f"✓ Batch sizes MATCH: {info}")
        else:
            logger.warning(f"✗ Batch size MISMATCH: {info}")

        return (images, prompts, info)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBatch": CLIPTextEncodeBatch,
    "CLIPTextEncodeSequence": CLIPTextEncodeSequence,
    "ConditioningSelector": ConditioningSelector,
    "PromptDebugger": PromptDebugger,
    "BatchSizeChecker": BatchSizeChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "CLIPTextEncodeSequence": "CLIP Text Encode (Sequence)",
    "ConditioningSelector": "Conditioning Selector (by Index)",
    "PromptDebugger": "Prompt Debugger",
    "BatchSizeChecker": "Batch Size Checker",
}
