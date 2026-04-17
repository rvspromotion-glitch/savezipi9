import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AdvancedImageSave.DownloadZip",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "AdvancedImageSave") return;

        // ── onNodeCreated: add the download button ──────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            this.savedFiles = [];

            const downloadBtn = this.addWidget(
                "button",
                "download_zip_btn",
                "Download All as ZIP",
                () => { this._triggerDownload(); }
            );

            downloadBtn.disabled = true;
            this._downloadBtn = downloadBtn;
        };

        // ── Shared download logic ────────────────────────────────────────────
        nodeType.prototype._triggerDownload = async function () {
            if (!this.savedFiles || this.savedFiles.length === 0) {
                alert("No images saved yet. Run the workflow first!");
                return;
            }

            const btn          = this._downloadBtn;
            const totalFiles   = this.savedFiles.length;
            const originalText = btn?.name;

            try {
                if (btn) { btn.name = `Creating ZIP (${totalFiles} images)…`; btn.disabled = true; }

                const response = await fetch("/download_batch_zip", {
                    method:  "POST",
                    headers: { "Content-Type": "application/json" },
                    body:    JSON.stringify({ files: this.savedFiles }),
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                if (btn) btn.name = "Downloading ZIP…";

                const blob    = await response.blob();
                const sizeMB  = (blob.size / (1024 * 1024)).toFixed(2);
                const url     = window.URL.createObjectURL(blob);
                const a       = document.createElement("a");
                a.href        = url;
                a.download    = `comfyui_batch_${totalFiles}_images.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                console.log(`✓ ZIP download complete: ${totalFiles} images, ${sizeMB} MB`);
                if (btn) { btn.name = originalText; btn.disabled = false; }

            } catch (error) {
                console.error("Error downloading zip:", error);
                alert(`Error downloading zip file:\n${error.message}`);
                if (btn) { btn.name = originalText; btn.disabled = false; }
            }
        };

        // ── onExecuted: store files, enable button, auto-download if set ─────
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (!message?.saved_files) return;

            this.savedFiles = message.saved_files;

            if (this._downloadBtn) {
                this._downloadBtn.disabled = false;
            }

            if (message.auto_download?.[0]) {
                this._triggerDownload().catch(err =>
                    console.error("[AdvancedImageSave] auto-download failed:", err)
                );
            }
        };
    }
});
