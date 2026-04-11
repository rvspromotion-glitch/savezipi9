import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// DatasetCaptionSave – adds a "Download Dataset ZIP" button to the node
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "BatchUtilityI9.DatasetCaptionSave",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "DatasetCaptionSave") return;

        // ── onNodeCreated: add the download button ──────────────────────────
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            this._datasetFiles = [];
            this._datasetCount = 0;

            // Status label (read-only text)
            this._statusWidget = this.addWidget(
                "text",
                "dataset_status",
                "Run the workflow to save a dataset.",
                () => {},
                { serialize: false }
            );

            // Download button
            const btn = this.addWidget(
                "button",
                "download_dataset_btn",
                "⬇ Download Dataset ZIP",
                async () => {
                    if (!this._datasetFiles || this._datasetFiles.length === 0) {
                        alert("No dataset saved yet – run the workflow first!");
                        return;
                    }

                    const total        = this._datasetFiles.length;
                    const pairCount    = this._datasetCount;
                    const originalText = btn.name;

                    try {
                        btn.name     = `Building ZIP (${pairCount} pairs, ${total} files)…`;
                        btn.disabled = true;

                        const response = await fetch("/download_dataset_zip", {
                            method:  "POST",
                            headers: { "Content-Type": "application/json" },
                            body:    JSON.stringify({
                                files:    this._datasetFiles,
                                zip_name: "dataset.zip",
                            }),
                        });

                        if (!response.ok) {
                            const errText = await response.text();
                            throw new Error(`HTTP ${response.status}: ${errText}`);
                        }

                        btn.name = "Downloading…";

                        const blob    = await response.blob();
                        const sizeMB  = (blob.size / 1024 / 1024).toFixed(2);
                        const url     = URL.createObjectURL(blob);
                        const anchor  = document.createElement("a");
                        anchor.href     = url;
                        anchor.download = "dataset.zip";
                        document.body.appendChild(anchor);
                        anchor.click();
                        URL.revokeObjectURL(url);
                        document.body.removeChild(anchor);

                        console.log(`✓ Dataset ZIP downloaded: ${pairCount} pairs, ${sizeMB} MB`);
                        btn.name     = originalText;
                        btn.disabled = false;

                    } catch (err) {
                        console.error("Dataset ZIP error:", err);
                        alert(`ZIP download failed:\n${err.message}`);
                        btn.name     = originalText;
                        btn.disabled = false;
                    }
                }
            );

            btn.disabled = true;
            this._downloadBtn = btn;
        };

        // ── onExecuted: receive saved_files from the Python node ────────────
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (!message) return;

            if (message.saved_files) {
                this._datasetFiles = message.saved_files;
                this._datasetCount = (message.dataset_count?.[0]) ?? 0;

                const outputDir = message.output_dir?.[0] ?? "";
                const pairCount = this._datasetCount;

                // Update status label
                if (this._statusWidget) {
                    this._statusWidget.value =
                        `✓ ${pairCount} pairs saved → ${outputDir}`;
                }

                // Enable download button
                if (this._downloadBtn) {
                    this._downloadBtn.name     = `⬇ Download Dataset ZIP  (${pairCount} pairs)`;
                    this._downloadBtn.disabled = false;
                }
            }
        };
    },
});
