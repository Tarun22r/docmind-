/**
 * Handles the Documents page: file upload (click + drag/drop),
 * loading the bundled sample document, and deleting documents.
 */

document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const statusBox = document.getElementById("upload-status");
    const loadSampleBtn = document.getElementById("load-sample-btn");

    if (!uploadArea) return; // not on the documents page

    function setStatus(message, kind) {
        statusBox.hidden = false;
        statusBox.textContent = message;
        statusBox.className = "upload-status" + (kind ? ` upload-status--${kind}` : "");
    }

    async function uploadFile(file) {
        setStatus(`Uploading ${file.name}...`, "progress");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const data = await apiRequest("/api/documents/upload", {
                method: "POST",
                body: formData,
            });
            const doc = data.document;
            if (doc.status === "ready") {
                setStatus(`${doc.filename} — ${doc.chunk_count} chunks indexed. Ready.`, "success");
            } else {
                setStatus(`${doc.filename} — ${doc.error_message || "Processing failed."}`, "error");
            }
        } catch (err) {
            setStatus(err.message, "error");
        }
        window.location.reload();
    }

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("is-dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("is-dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("is-dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    if (loadSampleBtn) {
        loadSampleBtn.addEventListener("click", async () => {
            setStatus("Loading sample document...", "progress");
            try {
                const data = await apiRequest("/api/documents/load-sample", { method: "POST" });
                setStatus(`${data.document.filename} loaded and indexed.`, "success");
            } catch (err) {
                setStatus(err.message, "error");
            }
            window.location.reload();
        });
    }

    document.querySelectorAll(".delete-doc-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const documentId = btn.dataset.documentId;
            if (!confirm("Delete this document and its index? This cannot be undone.")) return;
            try {
                await apiRequest(`/api/documents/${documentId}`, { method: "DELETE" });
                window.location.reload();
            } catch (err) {
                alert(err.message);
            }
        });
    });
});
