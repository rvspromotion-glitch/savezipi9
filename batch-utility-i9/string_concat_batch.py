import logging

logger = logging.getLogger("StringConcatBatch")


class StringConcatBatch:
    """
    Concatenates a prefix string and/or suffix string onto every prompt in a
    batch list.  Designed to sit between a Gemini batch node (which outputs a
    list of STRING) and downstream nodes such as CLIPTextEncodeBatch.

    Example
    -------
    prefix    = "laying on bed, "
    prompts   = ["a woman smiling", "a woman looking left"]
    suffix    = ", she wears no makeup"
    separator = ""          (already handled inside prefix/suffix above)

    result    = ["laying on bed, a woman smiling, she wears no makeup",
                 "laying on bed, a woman looking left, she wears no makeup"]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "prefix": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "e.g.  laying on bed, ",
                }),
                "suffix": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "e.g.  , she wears no makeup",
                }),
                "separator": ("STRING", {
                    "multiline": False,
                    "default": ", ",
                    "tooltip": (
                        "Inserted between prefix→prompt and prompt→suffix "
                        "only when the adjacent part is non-empty. "
                        "Set to empty string if you already include spacing "
                        "inside prefix / suffix."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "concat"
    CATEGORY = "utils/text"

    # prompts arrives as a list; the optional scalars arrive as single values
    INPUT_IS_LIST = (True, False, False, False)
    OUTPUT_IS_LIST = (True,)

    # ------------------------------------------------------------------ #

    def concat(self, prompts, prefix="", suffix="", separator=", "):
        """
        Parameters
        ----------
        prompts   : list[str]  – one string per image from the batch node
        prefix    : str        – prepended to every prompt
        suffix    : str        – appended  to every prompt
        separator : str        – glue inserted between non-empty parts
        """
        prefix    = prefix    or ""
        suffix    = suffix    or ""
        separator = separator if separator is not None else ", "

        results = []
        for idx, prompt in enumerate(prompts):
            parts = []
            if prefix:
                parts.append(prefix)
            parts.append(prompt)
            if suffix:
                parts.append(suffix)

            combined = separator.join(parts)
            results.append(combined)
            logger.debug(f"[{idx}] → {combined[:120]}")

        logger.info(f"✓ StringConcatBatch: processed {len(results)} prompts")
        return (results,)


# ------------------------------------------------------------------ #
# Registration
# ------------------------------------------------------------------ #

NODE_CLASS_MAPPINGS = {
    "StringConcatBatch": StringConcatBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConcatBatch": "String Concat (Batch)",
}
