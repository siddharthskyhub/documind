let currentMode = "hybrid";

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadDocuments();
  setupEventListeners();
});

function setupEventListeners() {
  // Mode tabs
  document.querySelectorAll(".mode-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-tab").forEach(b => {
        b.classList.remove("bg-sky-600", "text-white");
        b.classList.add("text-slate-400");
      });
      btn.classList.add("bg-sky-600", "text-white");
      btn.classList.remove("text-slate-400");
      currentMode = btn.dataset.mode;
    });
  });

  // Settings drawer toggle
  const toggleSettingsBtn = document.getElementById("toggleSettings");
  const settingsDrawer = document.getElementById("settingsDrawer");
  toggleSettingsBtn.addEventListener("click", () => {
    settingsDrawer.classList.toggle("hidden");
  });

  // Sliders
  document.getElementById("topKInput").addEventListener("input", (e) => {
    document.getElementById("topKVal").textContent = e.target.value;
  });
  document.getElementById("alphaInput").addEventListener("input", (e) => {
    document.getElementById("alphaVal").textContent = parseFloat(e.target.value).toFixed(2);
  });
  document.getElementById("tempInput").addEventListener("input", (e) => {
    document.getElementById("tempVal").textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Search trigger
  document.getElementById("searchBtn").addEventListener("click", executeQuery);
  document.getElementById("queryInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") executeQuery();
  });

  // Sample queries
  document.querySelectorAll(".sample-query").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById("queryInput").value = btn.textContent.trim();
      executeQuery();
    });
  });

  // Load samples button
  document.getElementById("loadSamplesBtn").addEventListener("click", loadSampleDocs);

  // File Upload Dropzone
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) uploadFile(e.target.files[0]);
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-sky-500", "bg-slate-900");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-sky-500", "bg-slate-900");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-sky-500", "bg-slate-900");
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });
}

async function checkHealth() {
  const badge = document.getElementById("statusBadge");
  const text = document.getElementById("statusText");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "healthy" && data.ollama.connected) {
      text.textContent = `Ollama Online (${data.indexed_chunks} Chunks)`;
      badge.className = "flex items-center space-x-2 text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400";
    } else {
      text.textContent = "Ollama Offline / Standby";
      badge.className = "flex items-center space-x-2 text-xs px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400";
    }
  } catch (err) {
    text.textContent = "Connecting to Backend...";
    badge.className = "flex items-center space-x-2 text-xs px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400";
  }
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    const list = document.getElementById("documentsList");
    const countBadge = document.getElementById("docCountBadge");

    countBadge.textContent = `${data.total_documents} docs (${data.total_chunks} chunks)`;

    if (data.documents.length === 0) {
      list.innerHTML = `
        <div class="text-center py-8 text-slate-500 text-xs">
          <i class="fa-regular fa-folder-open text-2xl mb-2 block"></i>
          No documents indexed yet.<br>Upload a file or click Load Samples!
        </div>`;
      return;
    }

    list.innerHTML = data.documents.map(doc => `
      <div class="p-3 bg-slate-950 border border-slate-800 rounded-xl flex items-center justify-between text-xs hover:border-slate-700 transition">
        <div class="truncate pr-2">
          <div class="font-semibold text-slate-200 truncate flex items-center gap-1.5">
            <i class="fa-regular fa-file-lines text-sky-400"></i>
            <span title="${doc.doc_name}">${doc.doc_name}</span>
          </div>
          <div class="text-[11px] text-slate-500 mt-0.5">
            ${doc.chunks} chunks &bull; ${doc.page_count} page(s)
          </div>
        </div>
        <button onclick="deleteDoc('${doc.doc_id}')" class="text-slate-500 hover:text-rose-400 p-1.5 rounded hover:bg-slate-900 transition" title="Delete document">
          <i class="fa-regular fa-trash-can"></i>
        </button>
      </div>
    `).join("");

  } catch (err) {
    console.error("Failed to load documents:", err);
  }
}

async function uploadFile(file) {
  const progress = document.getElementById("uploadProgress");
  progress.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      loadDocuments();
      checkHealth();
    } else {
      alert("Upload failed: " + (data.detail || "Unknown error"));
    }
  } catch (err) {
    alert("Upload error: " + err.message);
  } finally {
    progress.classList.add("hidden");
  }
}

async function loadSampleDocs() {
  const btn = document.getElementById("loadSamplesBtn");
  const orig = btn.innerHTML;
  btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Indexing Samples...`;
  btn.disabled = true;

  try {
    const res = await fetch("/api/load-samples", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      loadDocuments();
      checkHealth();
    }
  } catch (err) {
    alert("Failed to load samples: " + err.message);
  } finally {
    btn.innerHTML = orig;
    btn.disabled = false;
  }
}

async function deleteDoc(docId) {
  if (!confirm("Are you sure you want to delete this document from the knowledge base?")) return;
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    if (res.ok) {
      loadDocuments();
      checkHealth();
    }
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

async function executeQuery() {
  const query = document.getElementById("queryInput").value.trim();
  if (!query) return;

  const topK = parseInt(document.getElementById("topKInput").value);
  const alpha = parseFloat(document.getElementById("alphaInput").value);
  const temp = parseFloat(document.getElementById("tempInput").value);

  const loading = document.getElementById("queryLoading");
  const results = document.getElementById("resultsContainer");

  loading.classList.remove("hidden");
  results.classList.add("hidden");

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        mode: currentMode,
        top_k: topK,
        alpha: alpha,
        temperature: temp
      })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Query failed");

    renderResults(data);
  } catch (err) {
    alert("Error executing query: " + err.message);
  } finally {
    loading.classList.add("hidden");
  }
}

function renderResults(data) {
  const results = document.getElementById("resultsContainer");
  results.classList.remove("hidden");

  // Metrics
  document.getElementById("metricTotalTime").textContent = `${data.total_time_ms}ms`;
  document.getElementById("metricRetTime").textContent = `${data.retrieval_time_ms}ms`;
  document.getElementById("metricGenTime").textContent = `${data.generation_time_ms}ms`;
  document.getElementById("metricModeBadge").textContent = data.mode.replace("_", " ").toUpperCase();

  // Answer formatting with Markdown and interactive Citation Pills
  let formattedHtml = marked.parse(data.answer);
  // Replace [Ref: chunk_id] or [chunk_id] with clickable interactive pills
  formattedHtml = formattedHtml.replace(/\[(?:Ref:\s*)?([a-zA-Z0-9_\-]+)\]/g, (match, chunkId) => {
    return `<span class="citation-pill" onclick="highlightChunk('${chunkId}')"><i class="fa-solid fa-quote-right mr-1 text-[10px]"></i>Ref: ${chunkId}</span>`;
  });
  document.getElementById("answerContent").innerHTML = formattedHtml;

  // Grounding Faithfulness Badge
  const faithBadge = document.getElementById("faithfulnessBadge");
  const faithScore = document.getElementById("faithfulnessScore");
  const scorePercent = Math.round(data.grounding.faithfulness_score * 100);

  faithScore.textContent = `${scorePercent}% Grounded`;

  if (scorePercent >= 80) {
    faithBadge.className = "flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
    faithBadge.innerHTML = `<i class="fa-solid fa-shield-check"></i> <span>${scorePercent}% Grounded</span>`;
  } else if (scorePercent >= 60) {
    faithBadge.className = "flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30";
    faithBadge.innerHTML = `<i class="fa-solid fa-shield-halved"></i> <span>${scorePercent}% Partially Grounded</span>`;
  } else {
    faithBadge.className = "flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30";
    faithBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${scorePercent}% Potential Hallucination</span>`;
  }

  // Warnings
  const warningBar = document.getElementById("warningBar");
  const warningText = document.getElementById("warningText");
  if (data.grounding.warnings && data.grounding.warnings.length > 0) {
    warningText.innerHTML = data.grounding.warnings.map(w => `<div>&bull; ${w}</div>`).join("");
    warningBar.classList.remove("hidden");
  } else {
    warningBar.classList.add("hidden");
  }

  // Retrieved Evidence Chunks
  const chunksList = document.getElementById("chunksList");
  if (!data.sources || data.sources.length === 0) {
    chunksList.innerHTML = `<div class="text-xs text-slate-500 py-4 text-center">No chunks retrieved.</div>`;
    return;
  }

  chunksList.innerHTML = data.sources.map(chunk => {
    let highlightedText = escapeHtml(chunk.text);
    if (chunk.matched_keywords && chunk.matched_keywords.length > 0) {
      chunk.matched_keywords.forEach(kw => {
        const regex = new RegExp(`(${escapeRegExp(kw)})`, 'gi');
        highlightedText = highlightedText.replace(regex, `<span class="highlight-match">$1</span>`);
      });
    }

    const denseInfo = chunk.dense_rank !== null ? `Dense #${chunk.dense_rank} (${chunk.dense_score.toFixed(3)})` : `Dense: N/A`;
    const sparseInfo = chunk.sparse_rank !== null ? `BM25 #${chunk.sparse_rank} (${chunk.sparse_score.toFixed(2)})` : `BM25: N/A`;

    return `
      <div id="chunk-card-${chunk.chunk_id}" class="chunk-card bg-slate-900 border border-slate-800 rounded-xl p-4 transition duration-200">
        <div class="flex flex-wrap items-center justify-between gap-2 mb-2 pb-2 border-b border-slate-800/80">
          <div class="flex items-center gap-2">
            <span class="px-2 py-0.5 rounded font-mono font-bold text-xs bg-sky-500/10 text-sky-400 border border-sky-500/20">
              #${chunk.final_rank}
            </span>
            <span class="font-mono text-xs font-bold text-slate-200">${chunk.chunk_id}</span>
            <span class="text-xs text-slate-400">&bull; ${chunk.doc_name} (Page ${chunk.page})</span>
          </div>
          <div class="flex items-center gap-2 text-[11px] font-mono text-slate-400">
            <span class="bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-sky-300">RRF: ${chunk.rrf_score.toFixed(4)}</span>
            <span class="bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-indigo-300">${denseInfo}</span>
            <span class="bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-amber-300">${sparseInfo}</span>
          </div>
        </div>
        <div class="text-xs text-slate-300 leading-relaxed font-sans font-normal whitespace-pre-line">
          ${highlightedText}
        </div>
      </div>
    `;
  }).join("");
}

function highlightChunk(chunkId) {
  document.querySelectorAll(".chunk-card").forEach(c => c.classList.remove("chunk-highlight-active"));
  const target = document.getElementById(`chunk-card-${chunkId}`);
  if (target) {
    target.classList.add("chunk-highlight-active");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function escapeHtml(string) {
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
