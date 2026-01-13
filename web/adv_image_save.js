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
                        
                        try {
                            const response = await fetch("/download_batch_zip", {
                                method: "POST",
                                headers: {
                                    "Content-Type": "application/json"
                                },
                                body: JSON.stringify({
                                    files: this.savedFiles
                                })
                            });
                            
                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }
                            
                            // Download the zip file
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `comfyui_batch_${Date.now()}.zip`;
                            document.body.appendChild(a);
                            a.click();
                            window.URL.revokeObjectURL(url);
                            document.body.removeChild(a);
                            
                        } catch (error) {
                            console.error("Error downloading zip:", error);
                            alert("Error downloading zip file: " + error.message);
                        }
                    }
                );
                
                downloadBtn.disabled = true;
                this.savedFiles = [];
            };
        }
    }
});
