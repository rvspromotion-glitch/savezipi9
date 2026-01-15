import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AdvancedImageSave.DownloadZip",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AdvancedImageSave") {
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                // Store saved files data
                if (message.saved_files) {
                    this.savedFiles = message.saved_files;
                    
                    // Update widget if it exists
                    if (this.widgets) {
                        const widget = this.widgets.find(w => w.name === "download_zip_btn");
                        if (widget) {
                            widget.disabled = false;
                        }
                    }
                }
            };
            
            // Add download button widget after node is created
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const downloadBtn = this.addWidget(
                    "button",
                    "download_zip_btn",
                    "Download All as ZIP",
                    async () => {
                        if (!this.savedFiles || this.savedFiles.length === 0) {
                            alert("No images saved yet. Run the workflow first!");
                            return;
                        }

                        const totalFiles = this.savedFiles.length;
                        const originalText = downloadBtn.name;

                        try {
                            // Update button to show progress
                            downloadBtn.name = `Creating ZIP (${totalFiles} images)...`;
                            downloadBtn.disabled = true;

                            console.log(`Starting ZIP download for ${totalFiles} images...`);

                            const response = await fetch("/download_batch_zip", {
                                method: "POST",
                                headers: {
                                    "Content-Type": "application/json"
                                },
                                body: JSON.stringify({
                                    files: this.savedFiles
                                }),
                                // No timeout - let it take as long as needed
                            });

                            if (!response.ok) {
                                const errorText = await response.text();
                                throw new Error(`HTTP ${response.status}: ${errorText}`);
                            }

                            downloadBtn.name = "Downloading ZIP...";
                            console.log("ZIP created, downloading...");

                            // Download the zip file
                            const blob = await response.blob();
                            const sizeMB = (blob.size / (1024 * 1024)).toFixed(2);
                            console.log(`ZIP downloaded: ${sizeMB} MB`);

                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `comfyui_batch_${totalFiles}_images.zip`;
                            document.body.appendChild(a);
                            a.click();
                            window.URL.revokeObjectURL(url);
                            document.body.removeChild(a);

                            console.log(`✓ ZIP download complete: ${totalFiles} images, ${sizeMB} MB`);

                            // Reset button
                            downloadBtn.name = originalText;
                            downloadBtn.disabled = false;

                        } catch (error) {
                            console.error("Error downloading zip:", error);
                            alert(`Error downloading zip file:\n${error.message}\n\nCheck console logs for details.`);

                            // Reset button on error
                            downloadBtn.name = originalText;
                            downloadBtn.disabled = false;
                        }
                    }
                );
                
                downloadBtn.disabled = true;
                this.savedFiles = [];
            };
        }
    }
});
