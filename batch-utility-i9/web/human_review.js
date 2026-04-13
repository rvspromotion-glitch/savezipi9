/**
 * Human Review – batch image selector
 *
 * Listens for the "human_review_required" WebSocket event sent by
 * HumanReviewNode on its first pass, then:
 *
 *  1. Freezes every KSampler seed so ComfyUI's output cache stays valid
 *     when the workflow is re-queued (no upstream re-execution).
 *  2. Displays a full-screen modal grid of all batch images.
 *  3. Lets the user click images to select / deselect them (highlight ring).
 *  4. On "Continue", injects the comma-separated index string into the node's
 *     selected_indices widget, cleans up the server cache, and re-queues.
 *
 * After the second pass completes onExecuted resets selected_indices → ""
 * so the node is pristine for the next workflow run.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "BatchUtilityI9.HumanReview",

    /** Wire up the WebSocket listener once at load time. */
    async setup() {
        api.addEventListener("human_review_required", ({ detail }) => {
            const { execution_id, count } = detail ?? {};
            if (!execution_id || !count) return;
            showReviewModal(execution_id, count).catch(err =>
                console.error("[HumanReview] modal error:", err)
            );
        });
    },

    /** After a successful second-pass execution, reset the widget to ""
     *  so the next workflow run starts fresh on the first pass. */
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HumanReviewNode") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origOnExecuted?.apply(this, arguments);
            const w = this.widgets?.find(w => w.name === "selected_indices");
            if (w && w.value !== "") {
                console.log("[HumanReview] second pass done – resetting selected_indices");
                w.value = "";
            }
        };
    },
});

// ---------------------------------------------------------------------------
// Seed freeze helper
// ---------------------------------------------------------------------------

const SAMPLER_TYPES = new Set([
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "KSamplerSelect",
]);

function freezeAllSeeds() {
    const nodes = app.graph?._nodes ?? [];
    let frozen = 0;
    for (const node of nodes) {
        if (!node.widgets) continue;
        // Some nodes expose the sampler type in their type string
        const isSampler = SAMPLER_TYPES.has(node.type) ||
                          node.type?.toLowerCase().includes("sampler");
        if (!isSampler) continue;
        const ctrl = node.widgets.find(w => w.name === "control_after_generate");
        if (ctrl && ctrl.value !== "fixed") {
            ctrl.value = "fixed";
            frozen++;
        }
    }
    if (frozen > 0) {
        console.log(`[HumanReview] Froze control_after_generate on ${frozen} sampler node(s)`);
    }
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

async function showReviewModal(executionId, count) {
    // Freeze seeds before anything else so the cache is safe
    freezeAllSeeds();

    // ── Overlay backdrop ────────────────────────────────────────────────
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position:       "fixed",
        inset:          "0",
        background:     "rgba(0, 0, 0, 0.84)",
        zIndex:         "10000",
        display:        "flex",
        alignItems:     "center",
        justifyContent: "center",
        fontFamily:     "system-ui, sans-serif",
    });

    // ── Panel ───────────────────────────────────────────────────────────
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        background:    "#1a1a1a",
        border:        "1px solid #383838",
        borderRadius:  "12px",
        padding:       "24px",
        maxWidth:      "92vw",
        maxHeight:     "90vh",
        width:         "min(900px, 92vw)",
        display:       "flex",
        flexDirection: "column",
        gap:           "16px",
        color:         "#e0e0e0",
        boxShadow:     "0 24px 64px rgba(0,0,0,0.7)",
    });
    overlay.appendChild(panel);

    // ── Header ──────────────────────────────────────────────────────────
    const header = document.createElement("div");
    Object.assign(header.style, {
        display:        "flex",
        justifyContent: "space-between",
        alignItems:     "center",
        flexShrink:     "0",
    });

    const title = document.createElement("h2");
    Object.assign(title.style, {
        margin:   "0",
        fontSize: "17px",
        fontWeight: "600",
        color:    "#f0f0f0",
    });
    title.textContent = `Review batch  ·  ${count} image${count !== 1 ? "s" : ""}`;

    const closeBtn = document.createElement("button");
    Object.assign(closeBtn.style, {
        background: "transparent",
        border:     "none",
        color:      "#888",
        fontSize:   "20px",
        cursor:     "pointer",
        lineHeight: "1",
        padding:    "0 4px",
    });
    closeBtn.textContent = "✕";
    closeBtn.title = "Cancel – workflow stays paused";
    closeBtn.addEventListener("click", () => close());

    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    // ── Subtitle ────────────────────────────────────────────────────────
    const sub = document.createElement("p");
    Object.assign(sub.style, {
        margin:   "0",
        fontSize: "13px",
        color:    "#777",
        flexShrink: "0",
    });
    sub.textContent = "Click images to select them, then click Continue.";
    panel.appendChild(sub);

    // ── Image grid (scrollable) ─────────────────────────────────────────
    const gridWrap = document.createElement("div");
    Object.assign(gridWrap.style, {
        overflowY: "auto",
        flex:      "1 1 auto",
        minHeight: "0",
    });

    const grid = document.createElement("div");
    Object.assign(grid.style, {
        display:               "grid",
        gridTemplateColumns:   "repeat(auto-fill, minmax(148px, 1fr))",
        gap:                   "8px",
        padding:               "2px",   // keep focus rings visible
    });
    gridWrap.appendChild(grid);
    panel.appendChild(gridWrap);

    // ── Footer row ──────────────────────────────────────────────────────
    const footer = document.createElement("div");
    Object.assign(footer.style, {
        display:        "flex",
        justifyContent: "space-between",
        alignItems:     "center",
        flexShrink:     "0",
    });

    const hint = document.createElement("span");
    Object.assign(hint.style, { fontSize: "12px", color: "#555" });
    hint.textContent = "Closing without confirming leaves the workflow paused.";

    // ── Confirm button ──────────────────────────────────────────────────
    const confirmBtn = document.createElement("button");
    Object.assign(confirmBtn.style, {
        padding:      "10px 22px",
        borderRadius: "6px",
        border:       "none",
        fontSize:     "14px",
        fontWeight:   "600",
        cursor:       "not-allowed",
        background:   "#2a2a2a",
        color:        "#555",
        transition:   "background 0.15s, color 0.15s, transform 0.1s",
        minWidth:     "200px",
    });
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Continue with 0 selected";

    footer.appendChild(hint);
    footer.appendChild(confirmBtn);
    panel.appendChild(footer);

    // ── Selected set & update helpers ───────────────────────────────────
    const selected = new Set();

    function refreshConfirmBtn() {
        const n = selected.size;
        if (n === 0) {
            confirmBtn.disabled = true;
            Object.assign(confirmBtn.style, {
                background: "#2a2a2a",
                color:      "#555",
                cursor:     "not-allowed",
            });
            confirmBtn.textContent = "Continue with 0 selected";
        } else {
            confirmBtn.disabled = false;
            Object.assign(confirmBtn.style, {
                background: "#4ade80",
                color:      "#0a0a0a",
                cursor:     "pointer",
            });
            confirmBtn.textContent =
                `Continue with ${n} selected  (${count - n} discarded)`;
        }
    }

    // ── Build image tiles ───────────────────────────────────────────────
    for (let i = 0; i < count; i++) {
        const tile = document.createElement("div");
        Object.assign(tile.style, {
            position:     "relative",
            cursor:       "pointer",
            borderRadius: "6px",
            overflow:     "hidden",
            border:       "3px solid transparent",
            aspectRatio:  "1 / 1",
            background:   "#111",
            transition:   "border-color 0.12s, transform 0.1s",
            userSelect:   "none",
        });
        tile.setAttribute("tabindex", "0");
        tile.setAttribute("role", "checkbox");
        tile.setAttribute("aria-checked", "false");
        tile.setAttribute("aria-label", `Image ${i + 1}`);

        // Thumbnail
        const img = document.createElement("img");
        Object.assign(img.style, {
            width:      "100%",
            height:     "100%",
            objectFit:  "cover",
            display:    "block",
            transition: "opacity 0.1s",
        });
        img.src = `/human_review_image/${executionId}/${i}`;
        img.alt = `Image ${i + 1}`;
        img.draggable = false;
        img.onerror = () => {
            img.style.opacity    = "0.3";
            tile.style.background = "#300";
            label.textContent    = `${i + 1} ✗`;
        };

        // Index badge (bottom-left)
        const label = document.createElement("span");
        Object.assign(label.style, {
            position:   "absolute",
            bottom:     "5px",
            left:       "5px",
            background: "rgba(0,0,0,0.6)",
            color:      "#ddd",
            borderRadius: "4px",
            padding:    "1px 6px",
            fontSize:   "11px",
            fontWeight: "600",
            pointerEvents: "none",
        });
        label.textContent = i + 1;

        // Checkmark badge (top-right, hidden until selected)
        const check = document.createElement("div");
        Object.assign(check.style, {
            position:       "absolute",
            top:            "5px",
            right:          "5px",
            width:          "22px",
            height:         "22px",
            borderRadius:   "50%",
            background:     "#4ade80",
            color:          "#000",
            fontSize:       "13px",
            fontWeight:     "bold",
            display:        "none",
            alignItems:     "center",
            justifyContent: "center",
            pointerEvents:  "none",
            boxShadow:      "0 1px 4px rgba(0,0,0,0.4)",
        });
        check.textContent = "✓";

        tile.appendChild(img);
        tile.appendChild(label);
        tile.appendChild(check);
        grid.appendChild(tile);

        // Toggle on click or keyboard
        function toggle() {
            if (selected.has(i)) {
                selected.delete(i);
                tile.style.borderColor = "transparent";
                tile.style.transform   = "scale(1)";
                check.style.display    = "none";
                tile.setAttribute("aria-checked", "false");
            } else {
                selected.add(i);
                tile.style.borderColor = "#4ade80";
                tile.style.transform   = "scale(0.97)";
                check.style.display    = "flex";
                tile.setAttribute("aria-checked", "true");
            }
            refreshConfirmBtn();
        }

        tile.addEventListener("click", toggle);
        tile.addEventListener("keydown", e => {
            if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); }
        });
    }

    // ── Mount and close helpers ──────────────────────────────────────────
    document.body.appendChild(overlay);

    function close() {
        if (document.body.contains(overlay)) {
            document.body.removeChild(overlay);
        }
    }

    // Clicking outside the panel closes without action
    overlay.addEventListener("click", e => {
        if (e.target === overlay) close();
    });

    // ── Confirm action ───────────────────────────────────────────────────
    confirmBtn.addEventListener("click", async () => {
        if (selected.size === 0) return;

        const indices = [...selected].sort((a, b) => a - b).join(",");

        // Inject indices into the HumanReviewNode widget
        const reviewNode = app.graph?._nodes?.find(
            n => n.type === "HumanReviewNode"
        );
        if (reviewNode) {
            const w = reviewNode.widgets?.find(
                w => w.name === "selected_indices"
            );
            if (w) {
                w.value = indices;
            } else {
                console.warn("[HumanReview] 'selected_indices' widget not found on node");
            }
        } else {
            console.warn("[HumanReview] HumanReviewNode not found in graph");
        }

        // Free server-side image cache (fire-and-forget)
        fetch(`/human_review_cleanup/${executionId}`, { method: "POST" })
            .catch(() => {});

        close();

        // Re-queue – upstream output cache makes this instant
        try {
            await app.queuePrompt(0, 1);
        } catch (err) {
            console.error("[HumanReview] queuePrompt failed:", err);
            alert(
                "[Human Review] Could not re-queue automatically.\n" +
                "Please click the Queue Prompt button manually."
            );
        }
    });
}
