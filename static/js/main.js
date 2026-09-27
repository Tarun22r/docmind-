/**
 * Shared helpers used by upload.js and chat.js.
 */

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);
    let data = null;
    try {
        data = await response.json();
    } catch (err) {
        // Non-JSON response (e.g. a raw 500 HTML page) — fall through
        // with data left as null.
    }
    if (!response.ok) {
        const message = (data && data.error) ? data.error : `Request failed (${response.status}).`;
        throw new Error(message);
    }
    return data;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "";
    const mb = bytes / 1024 / 1024;
    return `${mb.toFixed(1)} MB`;
}
