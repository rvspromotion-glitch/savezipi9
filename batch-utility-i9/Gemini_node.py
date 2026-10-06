import concurrent.futures
import logging
import os
import random
import time

import google.generativeai as genai
from torch import Tensor

from . import openrouter
from .openrouter import OPENROUTER_INPUTS, OpenRouterError
from .utils import RETRY_INPUTS, GeminiGenerationError, generate_with_retry, images_to_pillow

class GeminiBatchNode:
    """
    Processes a batch of images through Gemini, generating one prompt per image.
    Output is a list of prompts matching the batch size.

    With provider=openrouter the images go to openrouter_model through
    OpenRouter instead, and after refusals_before_fallback refusals or failures
    to fallback_model. With provider=gemini and an OpenRouter key, Gemini gets
    refusals_before_fallback tries and fallback_model takes over after that.
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
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash-preview",
                        "gemini-3.1-flash-lite-preview",
                        "gemma-3-27b-it",
                        "gemma-3-12b-it",
                        "gemini-2.0-flash-001",
                        "gemini-2.0-flash-lite-001",
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
                **RETRY_INPUTS,
                **OPENROUTER_INPUTS,
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
        config_kwargs: dict,
        seed: int | None,
        max_retries: int,
        on_failure: str,
        proxy: str | None,
        batch_size: int,
        logger,
        route: dict | None = None,
    ):
        """
        Process a single image synchronously (runs in thread pool for concurrency).
        Retries with an incrementing seed; raises or falls back once retries run out.

        route: the OpenRouter models to use — {"key", "first", "models",
            "messages_kwargs", "chat_kwargs"}. "first" is True when Gemini is
            skipped; otherwise the models are the fallback after Gemini fails.

        Returns:
            tuple: (index, generated_prompt) to preserve ordering
        """
        label = f"Image {idx + 1}/{batch_size}"
        try:
            generated_prompt = None
            if not (route and route["first"]):
                try:
                    generated_prompt = generate_with_retry(
                        model_instance,
                        [prompt, pil_image],
                        config_kwargs,
                        seed=seed,
                        max_retries=route["gemini_tries"] if route else max_retries,
                        proxy=proxy,
                        logger=logger,
                        label=label,
                    )
                except GeminiGenerationError as e:
                    if not route:
                        raise
                    logger.warning(f"{e} – asking {route['models'][0][0]} instead")
            if generated_prompt is None:
                try:
                    generated_prompt = openrouter.chat_with_fallback(
                        route["key"],
                        route["models"],
                        openrouter.build_messages(prompt, [pil_image], **route["messages_kwargs"]),
                        seed=seed,
                        logger=logger,
                        label=label,
                        **route["chat_kwargs"],
                    )
                except OpenRouterError as e:
                    raise GeminiGenerationError(str(e)) from e
        except GeminiGenerationError as e:
            if on_failure == "stop":
                raise
            fallback_prompt = f"Error generating prompt for image {idx + 1}"
            logger.error(f"{e} – using fallback prompt: {fallback_prompt}")
            return (idx, fallback_prompt)

        # Detailed logging to debug
        logger.info(f"Image {idx + 1}/{batch_size}:")
        logger.info(f"  Generated prompt length: {len(generated_prompt)} characters")
        logger.info(f"  First 150 chars: {generated_prompt[:150]}...")
        logger.info(f"  Last 150 chars: ...{generated_prompt[-150:]}")
        logger.debug(f"  Full prompt: {generated_prompt}")

        return (idx, generated_prompt)

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
        max_retries: int = 5,
        on_failure: str = "stop",
        provider: str = "gemini",
        openrouter_api_key: str = "",
        openrouter_model: str = openrouter.DEFAULT_MODEL,
        fallback_model: str = openrouter.DEFAULT_FALLBACK_MODEL,
        refusals_before_fallback: int = 2,
    ):
        """Process each image in batch and return list of prompts."""

        logger = logging.getLogger("ComfyUI-Gemini-Batch")

        route = self._openrouter_route(
            provider, openrouter_api_key, openrouter_model, fallback_model,
            refusals_before_fallback, max_retries, system_instruction,
            response_type, temperature, num_predict, proxy,
        )

        model_instance = None
        if not (route and route["first"]):
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

        # Configure generation (seed is added per attempt by generate_with_retry)
        config_kwargs = dict(
            response_mime_type="application/json" if response_type == "json" else "text/plain",
            temperature=temperature,
        )
        if num_predict > 0:
            config_kwargs["max_output_tokens"] = num_predict

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
                    config_kwargs=config_kwargs,
                    seed=seed,
                    max_retries=max_retries,
                    on_failure=on_failure,
                    proxy=proxy,
                    batch_size=batch_size,
                    logger=logger,
                    route=route,
                )
                futures.append(future)

            # Wait for all futures to complete and collect results
            # Each result is a tuple: (index, generated_prompt)
            # With on_failure="stop" a failed image re-raises here and halts the run
            results = [future.result() for future in futures]

        # Sort results by index to ensure correct order
        # Even if image 8 finishes before image 5, this puts them back in order
        results.sort(key=lambda x: x[0])

        # Extract just the prompts in the correct order
        prompts = [prompt_text for idx, prompt_text in results]

        logger.info(f"Successfully generated {len(prompts)} prompts")
        logger.info(f"Prompt lengths: {[len(p) for p in prompts]}")
        return (prompts,)

    @staticmethod
    def _openrouter_route(
        provider, api_key, first_model, fallback_model, refusals_before_fallback,
        max_retries, system_instruction, response_type, temperature, num_predict, proxy,
    ):
        """Which OpenRouter models to ask, or None for Gemini only (the old behaviour)."""
        key = openrouter.resolve_key(api_key)
        fallback = (fallback_model or "").strip()
        tries_first = max(1, int(refusals_before_fallback))
        common = dict(
            key=key,
            messages_kwargs=dict(system_instruction=system_instruction or None),
            chat_kwargs=dict(
                temperature=temperature,
                max_tokens=num_predict if num_predict > 0 else None,
                json_mode=response_type == "json",
                proxy=proxy or None,
            ),
        )
        if provider == "openrouter":
            if not key:
                raise ValueError("provider=openrouter needs openrouter_api_key or OPENROUTER_API_KEY")
            first = (first_model or "").strip() or openrouter.DEFAULT_MODEL
            if fallback and fallback != first:
                models = [(first, tries_first), (fallback, max_retries)]
            else:
                models = [(first, max_retries)]
            return dict(common, first=True, models=models)
        if key and fallback:
            return dict(common, first=False, gemini_tries=tries_first, models=[(fallback, max_retries)])
        return None


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
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash-preview",
                        "gemini-3.1-flash-lite-preview",
                        "gemma-3-27b-it",
                        "gemma-3-12b-it",
                        "gemini-2.0-flash-001",
                        "gemini-2.0-flash-lite-001",
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
                "seed": ("INT", {"default": random.randint(1, 2**31), "min": 0, "max": 2**31, "step": 1}),
                **RETRY_INPUTS,
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
        config_kwargs: dict,
        seed: int | None,
        max_retries: int,
        on_failure: str,
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

        try:
            result_text = generate_with_retry(
                model_instance,
                [extraction_prompt] + pil_images,
                config_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logger,
                label="Carousel composition extraction",
            )
        except GeminiGenerationError as e:
            if on_failure == "stop":
                raise
            logger.error(f"{e} – using generic composition")
            return {
                "common": "wearing casual clothing",
                "per_image": ["standing in neutral pose"] * len(pil_images)
            }

        logger.info("Extracted carousel composition:")
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
        config_kwargs: dict,
        seed: int | None,
        max_retries: int,
        on_failure: str,
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

        try:
            generated_prompt = generate_with_retry(
                model_instance,
                [generation_prompt, pil_image],
                config_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logger,
                label=f"Image {idx + 1}/{batch_size}",
            )
        except GeminiGenerationError as e:
            if on_failure == "stop":
                raise
            fallback_prompt = f"{trigger_word} {common_composition}, {unique_composition}, {signature_features}, {style_suffix}"
            logger.error(f"{e} – using template fallback prompt for image {idx + 1}")
            return (idx, fallback_prompt)

        logger.info(f"Image {idx + 1}/{batch_size}:")
        logger.info(f"  Generated prompt length: {len(generated_prompt)} characters")
        logger.info(f"  First 200 chars: {generated_prompt[:200]}...")

        return (idx, generated_prompt)

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
        seed: int | None = None,
        max_retries: int = 5,
        on_failure: str = "stop",
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

        # Configure generation (seed is added per attempt by generate_with_retry)
        config_kwargs = dict(
            response_mime_type="text/plain",
            temperature=temperature,
        )
        if num_predict > 0:
            config_kwargs["max_output_tokens"] = num_predict
        retry_kwargs = dict(seed=seed, max_retries=max_retries, on_failure=on_failure)

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
            config_kwargs,
            proxy=proxy,
            logger=logger,
            **retry_kwargs,
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
                    config_kwargs=config_kwargs,
                    **retry_kwargs,
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


class GeminiDatasetBatchNode:
    """
    Like Ask Gemini (Batch) but sends requests one at a time with a
    configurable delay between each call.

    Use this instead of Ask Gemini (Batch) when you are hitting rate limits
    (429 errors) or when building large datasets where caption quality matters
    more than speed.

    request_interval – seconds to wait AFTER each successful request before
    sending the next one.  Start at 3 s for the free tier (15 RPM limit).
    Paid API keys can go lower (0.5 s is usually fine).
    """

    _MODEL_LIST = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemma-3-27b-it",
        "gemma-3-12b-it",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail for use as an image generation prompt.",
                    "multiline": True,
                }),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "response_type": (["text", "json"],),
                "model": (cls._MODEL_LIST,),
                "request_interval": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 0.5,
                    "tooltip": (
                        "Seconds to wait between each API request. "
                        "Free tier: 3 s (15 RPM limit). "
                        "Paid tier: 0.5–1 s is usually fine."
                    ),
                }),
                "min_chars": ("INT", {
                    "default": 150,
                    "min": 0,
                    "max": 5000,
                    "step": 25,
                    "tooltip": (
                        "Minimum acceptable caption length in characters. "
                        "If Gemini returns fewer characters it will retry "
                        "with an explicit length instruction appended. "
                        "Set to 0 to disable the check."
                    ),
                }),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "system_instruction": ("STRING", {}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_predict": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 1048576,
                    "step": 64,
                    "tooltip": (
                        "Max output tokens. 0 = unlimited. "
                        "If captions are cut off mid-sentence, raise this value. "
                        "512 is often too low; 1024 is a safe default for "
                        "detailed captions."
                    ),
                }),
                **RETRY_INPUTS,
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "process_batch"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Gemini/Batch"

    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_max_tokens(response) -> bool:
        """Return True when Gemini cut the response off at the token limit."""
        try:
            fr = response.candidates[0].finish_reason
            # The SDK exposes finish_reason as either an int or an enum.
            # Value 2 / name "MAX_TOKENS" means the output was truncated.
            return str(fr) in ("2", "MAX_TOKENS") or (
                hasattr(fr, "name") and fr.name == "MAX_TOKENS"
            )
        except (IndexError, AttributeError):
            return False

    def _call_once(self, idx, pil_image, model_instance, prompt, config_kwargs,
                   proxy, batch_size, logger, min_chars=0, seed=None,
                   max_retries=5, on_failure="stop"):
        """
        Single image → single caption.

        Empty responses and API errors are retried with an incrementing seed.
        On top of that, two soft failure modes are retried but the best
        result is kept if they never go away:
          1. MAX_TOKENS  – response was literally cut off mid-sentence because
                           num_predict is too low.  Logged clearly so the user
                           knows to raise the value.
          2. Too short   – response finished naturally but is under min_chars.
                           Retried with an explicit length instruction appended.
        """
        label = f"[{idx + 1}/{batch_size}]"

        def build_contents(last_caption):
            # On length-retry: tell the model it was too short
            if min_chars > 0 and last_caption is not None:
                active_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response was only "
                    f"{len(last_caption)} characters and appears incomplete. "
                    f"Write a complete, detailed description of at least "
                    f"{min_chars} characters. Do not stop early."
                )
            else:
                active_prompt = prompt
            return [active_prompt, pil_image]

        def check(caption, response):
            if self._is_max_tokens(response):
                return (
                    f"⚠ MAX_TOKENS hit – caption cut off at {len(caption)} chars. "
                    f"Raise num_predict (currently "
                    f"{config_kwargs.get('max_output_tokens', 'default')}) "
                    f"to fix mid-sentence truncation."
                )
            if min_chars > 0 and len(caption) < min_chars:
                return f"too short ({len(caption)} < {min_chars} chars)"
            return None

        try:
            caption = generate_with_retry(
                model_instance,
                build_contents,
                config_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logger,
                label=label,
                check=check,
            )
        except GeminiGenerationError as exc:
            if on_failure == "stop":
                raise
            logger.error(f"{exc} – using fallback caption")
            return f"Error generating caption for image {idx + 1}"

        logger.info(f"{label} OK – {len(caption)} chars  | {caption[:100]}…")
        return caption

    # ------------------------------------------------------------------ #

    def process_batch(
        self,
        images,
        prompt: str,
        safety_settings: str,
        response_type: str,
        model: str,
        request_interval: float = 3.0,
        min_chars: int = 150,
        api_key=None,
        proxy=None,
        system_instruction=None,
        seed=None,
        temperature: float = 0.7,
        num_predict: int = 512,
        max_retries: int = 5,
        on_failure: str = "stop",
    ):
        logger = logging.getLogger("ComfyUI-Gemini-Dataset-Batch")

        # Configure API
        if "GOOGLE_API_KEY" in os.environ and not api_key:
            genai.configure(transport="rest")
        else:
            genai.configure(api_key=api_key, transport="rest")

        model_instance = genai.GenerativeModel(
            model,
            safety_settings=safety_settings,
            system_instruction=system_instruction if system_instruction else None,
        )

        # Seed is added per attempt by generate_with_retry
        config_kwargs = dict(
            response_mime_type=(
                "application/json" if response_type == "json" else "text/plain"
            ),
            temperature=temperature,
        )
        if num_predict > 0:
            config_kwargs["max_output_tokens"] = num_predict

        pil_images = images_to_pillow(images)
        batch_size  = len(pil_images)

        logger.info(
            f"Dataset batch: {batch_size} images, "
            f"{request_interval}s interval, "
            f"min_chars={min_chars}"
        )

        captions = []
        for idx, pil_image in enumerate(pil_images):
            caption = self._call_once(
                idx, pil_image, model_instance, prompt,
                config_kwargs, proxy, batch_size, logger,
                min_chars=min_chars, seed=seed,
                max_retries=max_retries, on_failure=on_failure,
            )
            captions.append(caption)

            # Sleep AFTER each request except the last
            if idx < batch_size - 1 and request_interval > 0:
                logger.info(
                    f"Waiting {request_interval}s before next request "
                    f"({idx + 2}/{batch_size})…"
                )
                time.sleep(request_interval)

        logger.info(f"✓ Generated {len(captions)} captions")
        return (captions,)


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
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash-preview",
                        "gemini-3.1-flash-lite-preview",
                        "gemma-3-27b-it",
                        "gemma-3-12b-it",
                        "gemini-2.0-flash-001",
                        "gemini-2.0-flash-lite-001",
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
                # error_fallback_value already acts as on_failure here:
                # empty = stop the run, any text = use it as fallback
                "max_retries": RETRY_INPUTS["max_retries"],
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
        seed: int | None = None,
        max_retries: int = 5,
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
        config_kwargs = dict(
            response_mime_type="application/json" if response_type == "json" else "text/plain"
        )
        if temperature is not None and temperature >= 0:
            config_kwargs["temperature"] = temperature
        if num_predict is not None and num_predict > 0:
            config_kwargs["max_output_tokens"] = num_predict
        try:
            self.text_output = generate_with_retry(
                model,
                [prompt, *images_to_send],
                config_kwargs,
                seed=seed,
                max_retries=max_retries,
                proxy=proxy,
                logger=logging.getLogger("ComfyUI-Gemini"),
                label="Ask Gemini",
            )
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
    "Ask_Gemini_Dataset_Batch": GeminiDatasetBatchNode,
    "Ask_Gemini_Carousel_Character_Transfer": GeminiCarouselCharacterTransferNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini": "Ask Gemini",
    "Ask_Gemini_Batch": "Ask Gemini (Batch)",
    "Ask_Gemini_Dataset_Batch": "Ask Gemini (Dataset Batch)",
    "Ask_Gemini_Carousel_Character_Transfer": "Ask Gemini (Carousel + Character Transfer)",
}
