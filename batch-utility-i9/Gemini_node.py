import asyncio
import concurrent.futures
import logging
import os
import random

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

    async def _process_single_image(
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
        Process a single image asynchronously.

        Returns:
            tuple: (index, generated_prompt) to preserve ordering
        """
        try:
            # Use temporary_env_var context manager for proxy settings
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                # Call the async version of generate_content
                response = await model_instance.generate_content_async(
                    [prompt, pil_image],
                    generation_config=generation_config
                )
            generated_prompt = response.text.strip()

            # Detailed logging to debug
            logger.info(f"Image {idx + 1}/{batch_size}:")
            logger.info(f"  Generated prompt length: {len(generated_prompt)} characters")
            logger.info(f"  First 150 chars: {generated_prompt[:150]}...")
            logger.info(f"  Last 150 chars: ...{generated_prompt[-150:]}")
            logger.debug(f"  Full prompt: {generated_prompt}")

            return (idx, generated_prompt)

        except Exception as e:
            logger.error(f"Error processing image {idx + 1}/{batch_size}: {e}", exc_info=True)
            # Fallback to a generic prompt to maintain batch size
            fallback_prompt = f"Error generating prompt for image {idx + 1}"
            logger.warning(f"Using fallback prompt: {fallback_prompt}")
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

        # Create async tasks for all images
        # Each task will return (index, prompt) to preserve order
        async def process_all_images():
            tasks = []
            for idx, pil_image in enumerate(pil_images):
                task = self._process_single_image(
                    idx=idx,
                    pil_image=pil_image,
                    model_instance=model_instance,
                    prompt=prompt,
                    generation_config=generation_config,
                    proxy=proxy,
                    batch_size=batch_size,
                    logger=logger,
                )
                tasks.append(task)

            # Run all tasks concurrently and wait for all to complete
            # gather() returns results in the same order as tasks were provided
            results = await asyncio.gather(*tasks)
            return results

        # Run the async function and get results
        # results is a list of (index, prompt) tuples
        # Handle both cases: running event loop (ComfyUI) and no event loop
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # If we are, run in a thread pool to avoid nested loop issues
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.submit(
                    lambda: asyncio.run(process_all_images())
                ).result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            results = asyncio.run(process_all_images())

        # Sort results by index to ensure correct order
        # (This is extra safety, but gather() should already maintain order)
        results.sort(key=lambda x: x[0])

        # Extract just the prompts in the correct order
        prompts = [prompt_text for idx, prompt_text in results]

        logger.info(f"Successfully generated {len(prompts)} prompts")
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ask_Gemini": "Ask Gemini",
    "Ask_Gemini_Batch": "Ask Gemini (Batch)",
}
