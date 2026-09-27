/**
 * Handles the Chat page: document scope selection, sending questions,
 * and rendering answers with sources and an optional retrieved-context
 * viewer.
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    if (!form) return; // not on the chat page

    const input = document.getElementById("chat-input");
    const conversation = document.getElementById("conversation");
    const emptyState = document.getElementById("conversation-empty");
    const selectAll = document.getElementById("doc-select-all");
    const scopeCheckboxes = Array.from(document.querySelectorAll(".doc-scope-checkbox"));

    let conversationId = null;

    if (selectAll) {
        selectAll.addEventListener("change", () => {
            scopeCheckboxes.forEach((cb) => (cb.checked = selectAll.checked));
        });
        scopeCheckboxes.forEach((cb) => {
            cb.addEventListener("change", () => {
                selectAll.checked = scopeCheckboxes.every((c) => c.checked);
            });
        });
    }

    function getSelectedDocumentIds() {
        // An empty array tells the server "no scope filter" (i.e. search
        // across all documents), which is also correct when every
        // checkbox happens to be checked individually.
        if (!selectAll || selectAll.checked) return [];
        const selected = scopeCheckboxes.filter((cb) => cb.checked).map((cb) => parseInt(cb.value, 10));
        return selected;
    }

    function appendMessage(role, text) {
        if (emptyState) emptyState.remove();
        const wrapper = document.createElement("div");
        wrapper.className = `chat-message chat-message--${role}`;
        const bubble = document.createElement("div");
        bubble.className = "chat-message__bubble";
        bubble.textContent = text;
        wrapper.appendChild(bubble);
        conversation.appendChild(wrapper);
        conversation.scrollTop = conversation.scrollHeight;
        return wrapper;
    }

    function renderSources(wrapper, sources) {
        if (!sources || sources.length === 0) return;
        const box = document.createElement("div");
        box.className = "chat-sources";
        const title = document.createElement("div");
        title.className = "chat-sources__title";
        title.textContent = "Sources";
        box.appendChild(title);

        const list = document.createElement("ul");
        sources.forEach((s) => {
            const li = document.createElement("li");
            const page = s.page_number ? `Page ${s.page_number}` : "Page unavailable";
            li.textContent = `${s.document_name} — ${page}`;
            list.appendChild(li);
        });
        box.appendChild(list);
        wrapper.appendChild(box);
    }

    function renderRetrievedContext(wrapper, chunks) {
        if (!chunks || chunks.length === 0) return;

        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "context-toggle";
        toggle.textContent = "View retrieved context";

        const panel = document.createElement("div");
        panel.className = "retrieved-context";
        panel.hidden = true;

        chunks.forEach((chunk, i) => {
            const item = document.createElement("div");
            item.className = "retrieved-chunk";
            const meta = document.createElement("div");
            meta.className = "retrieved-chunk__meta";
            const page = chunk.page_number ? `Page ${chunk.page_number}` : "Page unavailable";
            meta.textContent = `Retrieved chunk ${i + 1} — ${chunk.document_name} — ${page} — similarity ${chunk.score}`;
            const text = document.createElement("div");
            text.textContent = chunk.content;
            item.appendChild(meta);
            item.appendChild(text);
            panel.appendChild(item);
        });

        toggle.addEventListener("click", () => {
            panel.hidden = !panel.hidden;
            toggle.textContent = panel.hidden ? "View retrieved context" : "Hide retrieved context";
        });

        wrapper.appendChild(toggle);
        wrapper.appendChild(panel);
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        appendMessage("user", query);
        input.value = "";
        input.disabled = true;

        const pending = appendMessage("assistant", "Thinking...");
        pending.classList.add("chat-message--pending");

        try {
            const data = await apiRequest("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    conversation_id: conversationId,
                    document_ids: getSelectedDocumentIds(),
                }),
            });

            conversationId = data.conversation_id;
            pending.classList.remove("chat-message--pending");
            pending.querySelector(".chat-message__bubble").textContent = data.answer;
            renderSources(pending, data.sources);
            renderRetrievedContext(pending, data.retrieved_chunks);
        } catch (err) {
            pending.classList.remove("chat-message--pending");
            pending.querySelector(".chat-message__bubble").textContent = `Error: ${err.message}`;
        } finally {
            input.disabled = false;
            input.focus();
        }
    });
});
