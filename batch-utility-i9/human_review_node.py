"""
HumanReviewNode
===============
Pauses a batch workflow mid-execution, lets the user cherry-pick images
via a frontend modal, then resumes with only the selected images.

How it works (two-pass execution)
----------------------------------
Pass 1  selected_indices == ""  (the default)
  • Converts the incoming IMAGE tensor to JPEG thumbnails, stores them in a
    module-level dict keyed by a fresh UUID (execution_id).
  • Sends a "human_review_required" WebSocket event to the frontend so the
    modal can open immediately.
  • Raises Exception("HUMAN_REVIEW_PENDING") to halt the workflow.

The frontend shows the grid, the user picks images and clicks "Continue".
The JS sets the node's selected_indices widget to e.g. "0,2,5", freezes
all KSampler seeds (so ComfyUI's output cache stays valid), then re-queues
the prompt.  ComfyUI replays from the review node forward — everything
upstream is served from cache so it's instant.

Pass 2  selected_indices == "0,2,5"
  • Parses the indices, slices the (cached) tensor, returns the filtered batch.
  • onExecuted fires in JS → resets the widget to "" ready for the next run.
"""

import io
import logging
import uuid

import numpy as np
import torch
from aiohttp import web
from PIL import Image
from server import PromptServer

logger = logging.getLogger("HumanReview")

# ---------------------------------------------------------------------------
# Module-level image cache
#   key   : execution_id (str UUID)
#   value : list[PIL.Image]   (one JPEG-quality thumbnail per batch image)
#
# We keep at most _MAX_PENDING sessions to avoid unbounded memory growth when
# users abandon reviews without confirming.
# ---------------------------------------------------------------------------
_pending_reviews: dict[str, list] = {}
_MAX_PENDING = 8


def _store_review(execution_id: str, pil_images: list) -> None:
    if len(_pending_reviews) >= _MAX_PENDING:
        oldest = next(iter(_pending_reviews))
        del _pending_reviews[oldest]
        logger.debug(f"Evicted oldest pending review to stay under limit ({_MAX_PENDING})")
    _pending_reviews[execution_id] = pil_images


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class HumanReviewNode:
    """
    Passthrough filter that pauses execution so the user can hand-pick which
    batch images continue downstream.

    enabled = False  →  instant passthrough, no interruption (useful for
                        bypassing the check during automated runs).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                # Populated automatically by the JS frontend on the second pass.
                # Do not wire this to another node — leave it as a plain widget.
                "selected_indices": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Leave empty. The review UI fills this in automatically "
                        "before re-queuing (e.g. '0,2,5')."
                    ),
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    FUNCTION      = "review"
    CATEGORY      = "image/batch"
    # NOT OUTPUT_NODE — this is a passthrough filter, not a terminal sink.

    # ------------------------------------------------------------------

    def review(self, images, enabled=True, selected_indices=""):
        # Defensive unwrap (ComfyUI can wrap widget values in lists)
        if isinstance(enabled, list):
            enabled = enabled[0] if enabled else True
        if isinstance(selected_indices, list):
            selected_indices = selected_indices[0] if selected_indices else ""

        # ── Passthrough mode ──────────────────────────────────────────────
        if not enabled:
            return (images,)

        selected_indices = (selected_indices or "").strip()

        # ── FIRST PASS: pause and open the review UI ──────────────────────
        if not selected_indices:
            execution_id = str(uuid.uuid4())
            batch_size   = images.shape[0]

            # Convert tensor → PIL images for the HTTP preview route
            pil_images = []
            for i in range(batch_size):
                arr = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr))
            _store_review(execution_id, pil_images)

            logger.info(
                f"Human review requested: {batch_size} images "
                f"(execution_id={execution_id})"
            )

            # Notify the frontend — this fires BEFORE we raise so the modal
            # can open even though the workflow is about to be interrupted.
            PromptServer.instance.send_sync(
                "human_review_required",
                {
                    "execution_id": execution_id,
                    "count":        batch_size,
                },
            )

            raise Exception("HUMAN_REVIEW_PENDING")

        # ── SECOND PASS: filter by selected indices ───────────────────────
        raw = [p.strip() for p in selected_indices.split(",") if p.strip()]
        try:
            indices = [int(x) for x in raw if x.isdigit()]
        except ValueError as exc:
            logger.error(f"Could not parse selected_indices '{selected_indices}': {exc}")
            return (images,)

        if not indices:
            logger.warning("No valid indices — returning full batch unchanged")
            return (images,)

        # Clamp to valid range and deduplicate while preserving order
        seen   = set()
        valid  = []
        for i in indices:
            if 0 <= i < images.shape[0] and i not in seen:
                valid.append(i)
                seen.add(i)

        if len(valid) != len(indices):
            dropped = set(indices) - set(valid)
            logger.warning(f"Dropped out-of-range indices: {sorted(dropped)}")

        filtered = torch.stack([images[i] for i in valid])
        logger.info(
            f"✓ Human review: {len(valid)}/{images.shape[0]} images selected "
            f"({valid})"
        )
        return (filtered,)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@PromptServer.instance.routes.get("/human_review_image/{execution_id}/{index}")
async def serve_review_image(request):
    """Serve one cached review image as JPEG."""
    execution_id = request.match_info["execution_id"]
    try:
        index = int(request.match_info["index"])
    except ValueError:
        return web.Response(status=400, text="index must be an integer")

    images = _pending_reviews.get(execution_id)
    if images is None:
        return web.Response(status=404, text="Review session not found or already cleaned up")
    if index < 0 or index >= len(images):
        return web.Response(status=404, text=f"Index {index} out of range (count={len(images)})")

    buf = io.BytesIO()
    images[index].save(buf, format="JPEG", quality=82)
    buf.seek(0)
    return web.Response(
        body=buf.read(),
        content_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@PromptServer.instance.routes.post("/human_review_cleanup/{execution_id}")
async def cleanup_review(request):
    """Called by the frontend after re-queuing to free cached images."""
    execution_id = request.match_info["execution_id"]
    if execution_id in _pending_reviews:
        del _pending_reviews[execution_id]
        logger.info(f"Cleaned up review session {execution_id}")
    return web.Response(status=200, text="OK")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "HumanReviewNode": HumanReviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HumanReviewNode": "Human Review (Batch Filter)",
}
