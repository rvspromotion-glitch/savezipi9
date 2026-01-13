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

                        # result is typically [[cond, {"pooled_output": pooled}]]
                        if isinstance(result, list) and len(result) > 0:
                            if isinstance(result[0], list) and len(result[0]) == 2:
                                cond = result[0][0]
                                pooled = result[0][1].get("pooled_output") if isinstance(result[0][1], dict) else None
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

                # Method 2: Fallback to tokenize + encode_from_tokens
                if method_1_failed:
                    logger.debug(f"[{idx+1}/{batch_size}] Using tokenize + encode_from_tokens method")
                    tokens = clip_model.tokenize(prompt_text)
                    logger.debug(f"[{idx+1}/{batch_size}] Tokenized successfully, tokens type: {type(tokens)}")

                    result = clip_model.encode_from_tokens(tokens, return_pooled=True)
                    logger.debug(f"[{idx+1}/{batch_size}] encode_from_tokens returned: type={type(result)}")

                    # Unpack the result
                    if isinstance(result, tuple) and len(result) == 2:
                        cond, pooled = result
                        logger.debug(f"[{idx+1}/{batch_size}] Unpacked: cond type={type(cond)}, pooled type={type(pooled)}")
                    else:
                        logger.error(f"[{idx+1}/{batch_size}] Unexpected return format from encode_from_tokens: {result}")
                        raise ValueError(f"encode_from_tokens returned unexpected format: {type(result)}")

                # Critical validation: ensure we got valid tensors
                if cond is None:
                    raise ValueError(f"CLIP returned None for cond tensor (prompt {idx+1})")
                if pooled is None:
                    raise ValueError(f"CLIP returned None for pooled tensor (prompt {idx+1})")

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
        logger.debug(f"Concatenating {len(padded_cond_tensors)} cond tensors and {len(pooled_tensors)} pooled tensors")
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
    "PromptDebugger": PromptDebugger,
    "BatchSizeChecker": BatchSizeChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeBatch": "CLIP Text Encode (Batch)",
    "CLIPTextEncodeSequence": "CLIP Text Encode (Sequence)",
    "PromptDebugger": "Prompt Debugger",
    "BatchSizeChecker": "Batch Size Checker",
}
