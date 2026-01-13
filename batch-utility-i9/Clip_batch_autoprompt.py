import logging
import torch
import folder_paths


class StringListToString:
    """
    Convert a list of strings from Gemini into individual string outputs.
    Use this with multiple CLIP Text Encode nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strings": ("STRING", {"forceInput": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string"
    
    CATEGORY = "utils"

    def get_string(self, strings, index):
        """Get a single string from the list by index."""
        if isinstance(strings, list):
            if index < len(strings):
                return (strings[index],)
            else:
                return (f"Index {index} out of range (list has {len(strings)} items)",)
        return (str(strings),)


class CLIPTextEncodeMultiple:
    """
    Encode multiple text prompts at once.
    This is designed to work with the Gemini batch output.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "text_1": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True}),
                "text_2": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True}),
                "text_3": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True}),
                "text_4": ("STRING", {"multiline": True, "dynamicPrompts": False, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("cond_1", "cond_2", "cond_3", "cond_4")
    FUNCTION = "encode"
    
    CATEGORY = "conditioning"

    def encode(self, clip, text_1=None, text_2=None, text_3=None, text_4=None):
        """Encode up to 4 texts."""
        results = []
        for text in [text_1, text_2, text_3, text_4]:
            if text:
                tokens = clip.tokenize(text)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                results.append([[cond, {"pooled_output": pooled}]])
            else:
                # Return empty conditioning if not provided
                results.append(None)
        return tuple(results)


class ConditioningConcat:
    """
    Concatenate multiple conditioning tensors along the batch dimension.
    This creates a batched conditioning from separate inputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
            },
            "optional": {
                "conditioning_2": ("CONDITIONING",),
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "conditioning_5": ("CONDITIONING",),
                "conditioning_6": ("CONDITIONING",),
                "conditioning_7": ("CONDITIONING",),
                "conditioning_8": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"
    
    CATEGORY = "conditioning"

    def concat(self, conditioning_1, conditioning_2=None, conditioning_3=None, conditioning_4=None,
               conditioning_5=None, conditioning_6=None, conditioning_7=None, conditioning_8=None):
        """Concatenate conditioning tensors into a batch."""
        logger = logging.getLogger("ConditioningConcat")
        
        conditionings = [conditioning_1]
        for cond in [conditioning_2, conditioning_3, conditioning_4, conditioning_5, 
                     conditioning_6, conditioning_7, conditioning_8]:
            if cond is not None:
                conditionings.append(cond)
        
        # Extract tensors
        cond_tensors = []
        pooled_tensors = []
        
        for cond in conditionings:
            cond_tensors.append(cond[0][0])
            pooled_tensors.append(cond[0][1]["pooled_output"])
        
        # Concatenate along batch dimension
        batched_cond = torch.cat(cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"Concatenated {len(conditionings)} conditionings into batch: {batched_cond.shape}")
        
        return ([[batched_cond, {"pooled_output": batched_pooled}]],)


class PromptListProcessor:
    """
    Takes the list output from Gemini and properly formats it for CLIP encoding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt_list": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "process"
    
    CATEGORY = "conditioning"

    def process(self, clip, prompt_list):
        """Process list of prompts into batched conditioning."""
        logger = logging.getLogger("PromptListProcessor")
        
        # Handle list input
        if not isinstance(prompt_list, list):
            prompt_list = [prompt_list]
        
        logger.info(f"Processing {len(prompt_list)} prompts")
        logger.debug(f"First prompt preview: {str(prompt_list[0])[:100]}")
        
        # Encode each prompt
        cond_tensors = []
        pooled_tensors = []
        
        for idx, prompt in enumerate(prompt_list):
            tokens = clip.tokenize(str(prompt))
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_tensors.append(cond)
            pooled_tensors.append(pooled)
        
        # Batch them
        batched_cond = torch.cat(cond_tensors, dim=0)
        batched_pooled = torch.cat(pooled_tensors, dim=0)
        
        logger.info(f"Created conditioning batch: {batched_cond.shape}")
        
        return ([[batched_cond, {"pooled_output": batched_pooled}]],)


NODE_CLASS_MAPPINGS = {
    "StringListToString": StringListToString,
    "CLIPTextEncodeMultiple": CLIPTextEncodeMultiple,
    "ConditioningConcat": ConditioningConcat,
    "PromptListProcessor": PromptListProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringListToString": "String List to String",
    "CLIPTextEncodeMultiple": "CLIP Text Encode (Multiple)",
    "ConditioningConcat": "Conditioning Concat (Batch)",
    "PromptListProcessor": "Prompt List to Conditioning",
}
