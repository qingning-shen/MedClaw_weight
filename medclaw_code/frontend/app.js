const BACKEND_URL = "http://localhost:5000";

const chatContainer = document.getElementById("chat-container");
const userInput     = document.getElementById("user-input");
const sendBtn       = document.getElementById("send-btn");

// Auto-resize textarea
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
});

// Send on Enter (Shift+Enter for newline)
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
sendBtn.addEventListener("click", handleSend);

// ── Message builders ──────────────────────────────────────────────────────────

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="avatar">You</div>
    <div class="bubble-wrap">
      <div class="bubble">${escapeHtml(text)}</div>
    </div>`;
  chatContainer.appendChild(row);
  scrollBottom();
  return row;
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble-wrap">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>`;
  chatContainer.appendChild(row);
  scrollBottom();
  return row;
}

function replaceWithAssistantMessage(typingRow, answer, steps) {
  const stepsHtml = buildStepsHtml(steps);
  const toggleId  = "toggle-" + Date.now();
  const listId    = "list-"   + Date.now();

  typingRow.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble-wrap">
      <div class="bubble">${escapeHtml(answer)}</div>
      ${steps.length > 0 ? `
        <div class="steps-toggle" id="${toggleId}" onclick="toggleSteps('${toggleId}','${listId}')">
          <span class="arrow">&#9658;</span>
          ${steps.length} tool call${steps.length > 1 ? "s" : ""} used
        </div>
        <div class="steps-list" id="${listId}">${stepsHtml}</div>
      ` : ""}
    </div>`;
  scrollBottom();
}

function appendErrorMessage(msg) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="avatar">AI</div>
    <div class="bubble-wrap">
      <div class="error-bubble">${escapeHtml(msg)}</div>
    </div>`;
  chatContainer.appendChild(row);
  scrollBottom();
}

function buildStepsHtml(steps) {
  return steps.map((s, i) => {
    const inputStr  = typeof s.input === "object"
      ? JSON.stringify(s.input, null, 2)
      : String(s.input);
    const outputStr = String(s.output).trim().slice(0, 600)
      + (String(s.output).length > 600 ? "\n..." : "");
    return `
      <div class="step-item">
        <div class="step-header">
          <span class="step-tag">TOOL</span>
          <span>${escapeHtml(s.tool)}</span>
        </div>
        <div class="step-section">Input:</div>
        <div class="step-code">${escapeHtml(inputStr)}</div>
        <div class="step-section" style="margin-top:6px">Output:</div>
        <div class="step-code">${escapeHtml(outputStr)}</div>
      </div>`;
  }).join("");
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function toggleSteps(toggleId, listId) {
  const toggle = document.getElementById(toggleId);
  const list   = document.getElementById(listId);
  toggle.classList.toggle("open");
  list.classList.toggle("visible");
}

function scrollBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setLoading(on) {
  sendBtn.disabled = on;
  userInput.disabled = on;
}

// ── Core send handler ─────────────────────────────────────────────────────────

async function handleSend() {
  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = "";
  userInput.style.height = "auto";
  setLoading(true);

  appendUserMessage(text);
  const typingRow = appendTypingIndicator();

  try {
    const res = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    replaceWithAssistantMessage(typingRow, data.answer, data.steps || []);
  } catch (err) {
    typingRow.remove();
    appendErrorMessage("Error: " + err.message);
  } finally {
    setLoading(false);
    userInput.focus();
  }
}
