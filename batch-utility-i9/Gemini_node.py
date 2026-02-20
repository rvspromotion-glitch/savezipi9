import concurrent.futures
import logging
import os
import random
import time

import google.generativeai as genai
from torch import Tensor

from .utils import images_to_pillow, temporary_env_var

class GeminiBatchNode:
    """
    Processes a batch of images through Gemini, generating one prompt per image.
    Output is a list of prompts matching the batch size.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "images": ("IMAGE",),  # Batch of images
                "prompt": ("STRING", {
                    "default": "Describe this image in detail for use as an image generation prompt.", 
                    "multiline": True
                }),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "response_type": (["text", "json"],),
                "model": (
                    [
                        "gemma-3-12b-it",
                        "gemma-3-27b-it",
                        "gemini-3-flash-preview",
                        "gemini-3-pro-preview",
                        "gemini-3-flash",
                        "gemini-3-pro",
                        "gemini-2.0-flash-lite-001",
                        "gemini-2.0-flash-001",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ],
                ),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "system_instruction": ("STRING", {}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_predict": ("INT", {"default": 512, "min": 0, "max": 1048576, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "process_batch"
    OUTPUT_IS_LIST = (True,)  # Critical: Output is a list of strings

    CATEGORY = "Gemini/Batch"

    def _process_single_image(
        self,
        idx: int,
        pil_image,
        model_instance,
        prompt: str,
        generation_config,
        proxy: str | None,
        batch_size: int,
        logger,
    ):
        """
        Process a single image synchronously (runs in thread pool for concurrency).
        Retries up to 3 times before falling back to error message.

        Returns:
            tuple: (index, generated_prompt) to preserve ordering
        """
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Use temporary_env_var context manager for proxy settings
                with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                    # Call the synchronous generate_content (thread-safe)
                    response = model_instance.generate_content(
                        [prompt, pil_image],
                        generation_config=generation_config
                    )
                generated_prompt = response.text.strip()

                # Detailed logging to debug
                logger.info(f"Image {idx + 1}/{batch_size} (attempt {attempt}):")
                logger.info(f"  Generated prompt length: {len(generated_prompt)} characters")
                logger.info(f"  First 150 chars: {generated_prompt[:150]}...")
                logger.info(f"  Last 150 chars: ...{generated_prompt[-150:]}")
                logger.debug(f"  Full prompt: {generated_prompt}")

                return (idx, generated_prompt)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Not the last attempt, log and retry
                    logger.warning(f"Image {idx + 1}/{batch_size} attempt {attempt} failed: {e}")
                    logger.info(f"Retrying image {idx + 1}/{batch_size} (attempt {attempt + 1}/{max_retries})...")
                    # Short delay before retry to avoid rate limiting
                    time.sleep(0.5)
                else:
                    # Last attempt failed, log full error
                    logger.error(f"Image {idx + 1}/{batch_size} failed after {max_retries} attempts: {e}", exc_info=True)

        # All retries exhausted, use fallback
        fallback_prompt = f"Error generating prompt for image {idx + 1}"
        logger.warning(f"Using fallback prompt after {max_retries} failed attempts: {fallback_prompt}")
        return (idx, fallback_prompt)

    def process_batch(
        self,
        images: Tensor,
        prompt: str,
        safety_settings: str,
        response_type: str,
        model: str,
        api_key: str | None = None,
        proxy: str | None = None,
        system_instruction: str | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
        num_predict: int = 512,
    ):
        """Process each image in batch and return list of prompts."""
        
        logger = logging.getLogger("ComfyUI-Gemini-Batch")
        
        # Configure API
        if "GOOGLE_API_KEY" in os.environ and not api_key:
            genai.configure(transport="rest")
        else:
            genai.configure(api_key=api_key, transport="rest")
        
        # Initialize model
        model_instance = genai.GenerativeModel(
            model, 
            safety_settings=safety_settings, 
            system_instruction=system_instruction if system_instruction else None
        )
        
        # Configure generation
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json" if response_type == "json" else "text/plain",
            temperature=temperature,
        )
        if num_predict > 0:
            generation_config.max_output_tokens = num_predict
        
        # Convert batch tensor to list of PIL images
        pil_images = images_to_pillow(images)
        batch_size = len(pil_images)

        logger.info(f"Processing batch of {batch_size} images through Gemini (concurrently)")

        # Use ThreadPoolExecutor to process all images concurrently
        # Each thread will make a blocking call to Gemini API
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks to the thread pool
            # Each returns a Future object
            futures = []
            for idx, pil_image in enumerate(pil_images):
                future = executor.submit(
                    self._process_single_image,
                    idx=idx,
                    pil_image=pil_image,
                    model_instance=model_instance,
                    prompt=prompt,
                    generation_config=generation_config,
                    proxy=proxy,
                    batch_size=batch_size,
                    logger=logger,
                )
                futures.append(future)

            # Wait for all futures to complete and collect results
            # Each result is a tuple: (index, generated_prompt)
            results = [future.result() for future in futures]

        # Sort results by index to ensure correct order
        # Even if image 8 finishes before image 5, this puts them back in order
        results.sort(key=lambda x: x[0])

        # Extract just the prompts in the correct order
        prompts = [prompt_text for idx, prompt_text in results]

        logger.info(f"Successfully generated {len(prompts)} prompts")
        logger.info(f"Prompt lengths: {[len(p) for p in prompts]}")
        return (prompts,)


class GeminiCarouselCharacterTransferNode:
    """
    Carousel + Character Transfer - the ultimate combo node.

    Use case: Reference images show Person A in consistent outfit/style across different
    settings. You want to generate prompts for YOUR character (Person B) wearing the
    SAME outfit/style in those SAME settings.

    Example:
      Reference: 4 photos of Instagram model in same red dress, different locations
      Your LoRA: "3lm1ra" character
      Output: 4 prompts of 3lm1ra in that red dress at those locations

    Two-phase process:
    1. Extract COMMON + UNIQUE composition from all reference images
       - COMMON: Outfit, style, accessories, vibe (what stays same across all images)
       - UNIQUE: Pose, setting, lighting per image (what varies)
       - IGNORES: The person's actual appearance
    2. Generate prompts: YOUR character + common elements + unique per-image details
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (
                    [
                        "gemma-3-12b-it",
                        "gemma-3-27b-it",
                        "gemini-3-flash-preview",
                        "gemini-3-pro-preview",
                        "gemini-3-flash",
                        "gemini-3-pro",
                        "gemini-2.0-flash-lite-001",
                        "gemini-2.0-flash-001",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ],
                ),
                "trigger_word": ("STRING", {
                    "default": "3lm1ra, light-skinned woman with long straight blonde balayage hair with dark roots, grey eyes and only a little bit of freckles on cheeks.",
                    "multiline": True
                }),
                "signature_features": ("STRING", {
                    "default": "featuring sharp dark defined eyebrows, dramatic long wispy Russian-volume false lashes, full-coverage matte foundation with heavy contour, and lips heavily overlined with dark liner and high-shine glass gloss",
                    "multiline": True
                }),
                "style_suffix": ("STRING", {
                    "default": "Instagirl, kept delicate noise texture, dangerous charm, amateur cellphone quality, visible sensor noise, heavy HDR glow, amateur photo, blown-out highlight from the lamp, deeply crushed shadows.",
                    "multiline": True
                }),
            },
            "optional": {
                "body_highlight_template": ("STRING", {
                    "default": "explicitly highlighting her big bust, small waist, and big ass",
                    "multiline": False
                }),
                "eye_highlight_template": ("STRING", {
                    "default": "explicitly highlighting her {adjective} grey eyes",
                    "multiline": False
                }),
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "system_instruction": ("STRING", {}),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"], {"default": "BLOCK_NONE"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_predict": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "process_carousel_character_transfer"
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "Gemini/Batch"

    def _extract_carousel_composition(
        self,
        pil_images,
        model_instance,
        generation_config,
        proxy: str | None,
        logger,
    ):
        """
        Phase 1: Extract COMMON and UNIQUE composition elements.

        COMMON: What stays the same across ALL images (outfit, accessories, style, vibe)
        UNIQUE: What varies per image (pose, setting, lighting)

        IGNORES: The person's actual appearance (face, hair color, body, etc.)

        Returns:
            dict: {"common": str, "per_image": list[str]}
        """
        extraction_prompt = (
            "Analyze these reference images showing the same person in different settings.\n\n"
            "Your task is to extract TWO types of information, COMPLETELY IGNORING the person's "
            "actual appearance (face, hair color, eye color, skin tone, body type, etc.):\n\n"
            "PART 1 - COMMON ELEMENTS (what stays CONSISTENT across ALL images):\n"
            "- OUTFIT & CLOTHING: Specific garments, colors, patterns, textures, logos, text on clothing\n"
            "- ACCESSORIES & JEWELRY: Necklaces, earrings, rings, watches, bags, etc.\n"
            "- NAILS (if visible): Shape, finish, design that's consistent\n"
            "- OVERALL STYLE/VIBE: Aesthetic (glam, edgy, casual, etc.)\n\n"
            "PART 2 - UNIQUE ELEMENTS PER IMAGE (what VARIES between images):\n"
            "For each image separately:\n"
            "- POSE & BODY LANGUAGE: Exact position, gestures, interaction with environment\n"
            "- SETTING & BACKGROUND: Specific location, props, furniture\n"
            "- LIGHTING: Light source, direction, intensity\n\n"
            "Format your response EXACTLY like this:\n"
            "COMMON ELEMENTS:\n"
            "[description of outfit, accessories, style that appears in ALL images]\n\n"
            "IMAGE 1:\n"
            "[specific pose, setting, lighting for image 1]\n\n"
            "IMAGE 2:\n"
            "[specific pose, setting, lighting for image 2]\n\n"
            "etc.\n\n"
            "CRITICAL: Do NOT describe the person's face, hair color, eye color, skin, or body shape."
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                    content_list = [extraction_prompt] + pil_images
                    response = model_instance.generate_content(
                        content_list,
                        generation_config=generation_config
                    )
                result_text = response.text.strip()

                logger.info(f"Extracted carousel composition (attempt {attempt}):")
                logger.info(f"  {result_text[:500]}...")

                # Parse the response
                common_section = ""
                per_image = []

                lines = result_text.split('\n')
                current_section = None
                current_content = []

                for line in lines:
                    line_upper = line.strip().upper()
                    if line_upper.startswith("COMMON ELEMENTS"):
                        current_section = "common"
                        current_content = []
                    elif line_upper.startswith("IMAGE "):
                        if current_section == "common":
                            common_section = '\n'.join(current_content).strip()
                        elif current_section == "image":
                            per_image.append('\n'.join(current_content).strip())
                        current_section = "image"
                        current_content = []
                    elif line.strip():
                        current_content.append(line)

                # Add last section
                if current_section == "image":
                    per_image.append('\n'.join(current_content).strip())

                if not common_section:
                    common_section = "wearing a stylish outfit"
                if not per_image:
                    per_image = ["standing in a neutral pose"] * len(pil_images)

                return {"common": common_section, "per_image": per_image}

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Carousel composition extraction attempt {attempt} failed: {e}")
                    logger.info(f"Retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(0.5)
                else:
                    logger.error(f"Carousel composition extraction failed after {max_retries} attempts: {e}", exc_info=True)
                    return {
                        "common": "wearing casual clothing",
                        "per_image": ["standing in neutral pose"] * len(pil_images)
                    }

    def _generate_carousel_character_prompt(
        self,
        idx: int,
        pil_image,
        common_composition: str,
        unique_composition: str,
        trigger_word: str,
        signature_features: str,
        body_highlight: str,
        eye_highlight: str,
        style_suffix: str,
        model_instance,
        generation_config,
        proxy: str | None,
        batch_size: int,
        logger,
    ):
        """
        Phase 2: Generate final prompt combining:
        - YOUR character template
        - COMMON composition (shared outfit/style)
        - UNIQUE composition (this image's specific pose/setting)

        Returns:
            tuple: (index, generated_prompt) to preserve ordering
        """
        generation_prompt = (
            f"You are a professional prompt engineer for Flux/Stable Diffusion.\n\n"
            f"CHARACTER TEMPLATE (use this EXACTLY):\n"
            f"Trigger: {trigger_word}\n"
            f"Signature Features: {signature_features}\n"
            f"Body Highlight (use if body visible): {body_highlight}\n"
            f"Eye Highlight (use if eyes visible): {eye_highlight}\n"
            f"Style Suffix: {style_suffix}\n\n"
            f"COMMON ELEMENTS (consistent across all images in this carousel):\n"
            f"{common_composition}\n\n"
            f"UNIQUE ELEMENTS (specific to THIS image):\n"
            f"{unique_composition}\n\n"
            f"Your task:\n"
            f"1. Analyze this reference image to understand the exact details\n"
            f"2. Generate a detailed Flux/SD prompt that describes the CHARACTER TEMPLATE "
            f"wearing/styled with the COMMON ELEMENTS in the UNIQUE ELEMENTS composition\n"
            f"3. Start with trigger word, include signature features, add body/eye highlights if applicable\n"
            f"4. Describe the outfit/accessories from COMMON ELEMENTS in detail (logos, text, materials, colors)\n"
            f"5. Describe the pose, setting, and lighting from UNIQUE ELEMENTS\n"
            f"6. Include specific details: brand names, nail design if visible, props, lighting physics\n"
            f"7. End with the style suffix\n\n"
            f"Output ONLY the final prompt - no explanations, no metadata, just the prompt text."
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                    response = model_instance.generate_content(
                        [generation_prompt, pil_image],
                        generation_config=generation_config
                    )
                generated_prompt = response.text.strip()

                logger.info(f"Image {idx + 1}/{batch_size} (attempt {attempt}):")
                logger.info(f"  Generated prompt length: {len(generated_prompt)} characters")
                logger.info(f"  First 200 chars: {generated_prompt[:200]}...")

                return (idx, generated_prompt)

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Image {idx + 1}/{batch_size} attempt {attempt} failed: {e}")
                    logger.info(f"Retrying image {idx + 1}/{batch_size} (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(0.5)
                else:
                    logger.error(f"Image {idx + 1}/{batch_size} failed after {max_retries} attempts: {e}", exc_info=True)

        fallback_prompt = f"{trigger_word} {common_composition}, {unique_composition}, {signature_features}, {style_suffix}"
        logger.warning(f"Using fallback prompt for image {idx + 1}")
        return (idx, fallback_prompt)

    def process_carousel_character_transfer(
        self,
        images: Tensor,
        model: str,
        trigger_word: str,
        signature_features: str,
        style_suffix: str,
        body_highlight_template: str = "explicitly highlighting her big bust, small waist, and big ass",
        eye_highlight_template: str = "explicitly highlighting her {adjective} grey eyes",
        api_key: str | None = None,
        proxy: str | None = None,
        system_instruction: str | None = None,
        safety_settings: str = "BLOCK_NONE",
        temperature: float = 0.7,
        num_predict: int = 1024,
    ):
        """
        Process carousel of reference images and transfer to your character with consistency.
        """

        logger = logging.getLogger("ComfyUI-Gemini-CarouselCharacterTransfer")

        # Configure API
        if "GOOGLE_API_KEY" in os.environ and not api_key:
            genai.configure(transport="rest")
        else:
            genai.configure(api_key=api_key, transport="rest")

        # Initialize model
        model_instance = genai.GenerativeModel(
            model,
            safety_settings=safety_settings,
            system_instruction=system_instruction if system_instruction else None
        )

        # Configure generation
        generation_config = genai.GenerationConfig(
            response_mime_type="text/plain",
            temperature=temperature,
        )
        if num_predict > 0:
            generation_config.max_output_tokens = num_predict

        # Convert batch tensor to list of PIL images
        pil_images = images_to_pillow(images)
        batch_size = len(pil_images)

        logger.info(f"Processing carousel character transfer for {batch_size} images")
        logger.info(f"Character: {trigger_word[:50]}...")

        # PHASE 1: Extract common + unique composition elements
        logger.info("Phase 1: Extracting carousel composition (common + unique per image)...")
        composition = self._extract_carousel_composition(
            pil_images,
            model_instance,
            generation_config,
            proxy,
            logger,
        )

        common_elements = composition["common"]
        per_image_elements = composition["per_image"]

        logger.info(f"Common elements: {common_elements[:200]}...")
        logger.info(f"Got {len(per_image_elements)} unique image descriptions")

        # PHASE 2: Generate character prompts maintaining carousel consistency
        logger.info(f"Phase 2: Generating {batch_size} carousel character prompts...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for idx, pil_image in enumerate(pil_images):
                # Get this image's unique composition, or use first one as fallback
                unique_comp = per_image_elements[idx] if idx < len(per_image_elements) else per_image_elements[0]

                future = executor.submit(
                    self._generate_carousel_character_prompt,
                    idx=idx,
                    pil_image=pil_image,
                    common_composition=common_elements,
                    unique_composition=unique_comp,
                    trigger_word=trigger_word,
                    signature_features=signature_features,
                    body_highlight=body_highlight_template,
                    eye_highlight=eye_highlight_template,
                    style_suffix=style_suffix,
                    model_instance=model_instance,
                    generation_config=generation_config,
                    proxy=proxy,
                    batch_size=batch_size,
                    logger=logger,
                )
                futures.append(future)

            results = [future.result() for future in futures]

        # Sort results by index to ensure correct order
        results.sort(key=lambda x: x[0])

        # Extract just the prompts in the correct order
        prompts = [prompt_text for idx, prompt_text in results]

        logger.info(f"Successfully generated {len(prompts)} carousel character transfer prompts")
        logger.info(f"Prompt lengths: {[len(p) for p in prompts]}")
        return (prompts,)


# Keep original single-image node for compatibility
class GeminiNode:
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "prompt": ("STRING", {"default": "Why number 42 is important?", "multiline": True}),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "response_type": (["text", "json"],),
                "model": (
                    [
                        "gemma-3-12b-it",
                        "gemma-3-27b-it",
                        "gemini-3-flash-preview",
                        "gemini-3-pro-preview",
                        "gemini-3-flash",
                        "gemini-3-pro",
                        "gemini-2.0-flash-lite-001",
                        "gemini-2.0-flash-001",
                        "gemini-2.5-flash",
                        "gemini-2.5-pro",
                    ],
                ),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "system_instruction": ("STRING", {}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "ask_gemini"

    CATEGORY = "Gemini"

    def __init__(self):
        self.text_output: str | None = None

    def ask_gemini(self, **kwargs):
        return (kwargs["error_fallback_value"] if self.text_output is None else self.text_output,)

    def check_lazy_status(
        self,
        prompt: str,
        safety_settings: str,
        response_type: str,
        model: str,
        api_key: str | None = None,
        proxy: str | None = None,
        image_1: Tensor | list[Tensor] | None = None,
        image_2: Tensor | list[Tensor] | None = None,
        image_3: Tensor | list[Tensor] | None = None,
        system_instruction: str | None = None,
        error_fallback_value: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        **kwargs,
    ):
        self.text_output = None
        if not system_instruction:
            system_instruction = None
        images_to_send = []
        for image in [image_1, image_2, image_3]:
            if image is not None:
                images_to_send.extend(images_to_pillow(image))
        if "GOOGLE_API_KEY" in os.environ and not api_key:
            genai.configure(transport="rest")
        else:
            genai.configure(api_key=api_key, transport="rest")
        model = genai.GenerativeModel(model, safety_settings=safety_settings, system_instruction=system_instruction)
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json" if response_type == "json" else "text/plain"
        )
        if temperature is not None and temperature >= 0:
            generation_config.temperature = temperature
        if num_predict is not None and num_predict > 0:
            generation_config.max_output_tokens = num_predict
        try:
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                response = model.generate_content([prompt, *images_to_send], generation_config=generation_config)
            self.text_output = response.text
        except Exception:
            if error_fallback_value is None:
                logging.getLogger("ComfyUI-Gemini").debug("ComfyUI-Gemini: exception occurred:", exc_info=True)
                return ["error_fallback_value"]
            if error_fallback_value == "":
                raise
        return []


NODE_CLASS_MAPPINGS = {
    "Ask_Gemini": GeminiNode,
    "Ask_Gemini_Batch": GeminiBatchNode,
    "Ask_Gemini_Carousel_Character_Transfer": GeminiCarouselCharacterTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini": "Ask Gemini",
    "Ask_Gemini_Batch": "Ask Gemini (Batch)",
    "Ask_Gemini_Carousel_Character_Transfer": "Ask Gemini (Carousel + Character Transfer)",
}
