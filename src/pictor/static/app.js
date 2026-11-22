// Pictor's browser UI. No framework and no build step: it talks to the local server's
// JSON API and renders with plain DOM calls.

const $ = (selector, root = document) => root.querySelector(selector);

/** Create an element. Text children are inserted as text, never as HTML. */
function h(tag, props = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") element.className = value;
    else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else if (value === true) element.setAttribute(key, "");
    else element.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    element.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return element;
}

const app = {
  status: null,
  session: null, // server state of the open picture
  transcript: [], // [{role: "user", text} | {role: "assistant", parts: []}]
  busy: false,
  controller: null,
  comparing: false,
  setup: { models: null, choice: null, custom: "", pull: null, error: "", timer: null },
};

const SUGGESTIONS = [
  "Remove the background",
  "Crop it to a square",
  "Make it black and white",
  "Blur the faces",
  "Describe this picture",
];

// ---------------------------------------------------------------- server calls

async function api(path, { method, json, body, signal, raw } = {}) {
  const init = { method: method || (json !== undefined || body ? "POST" : "GET"), signal };
  if (json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(json);
  } else if (body) {
    init.body = body;
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText || `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg).join("; ");
    } catch {
      // Not JSON; keep the status text.
    }
    throw new Error(detail);
  }
  return raw ? response : response.json();
}

/** Read a streamed response of one JSON object per line. */
async function* ndjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

function toast(message, kind = "") {
  const element = h("div", { class: `toast ${kind}` }, message);
  $("#toasts").append(element);
  setTimeout(() => element.remove(), kind === "error" ? 7000 : 3500);
}

const imageUrl = (versionId, width) =>
  `/api/session/${app.session.id}/image/${versionId}?w=${width}`;

// ---------------------------------------------------------------- text formatting

function escapeHtml(text) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return text.replace(/[&<>"']/g, (char) => entities[char]);
}

function inline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}

/** A deliberately tiny Markdown subset. Input is escaped first, so the only tags that
 *  can appear in the output are the ones added here. */
function markdown(source) {
  let html = "";
  let inList = false;
  for (const line of escapeHtml(source.trim()).split("\n")) {
    const item = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      if (!inList) html += "<ul>";
      inList = true;
      html += `<li>${inline(item[1])}</li>`;
      continue;
    }
    if (inList) html += "</ul>";
    inList = false;
    if (line.trim()) html += `<p>${inline(line.replace(/^#+\s*/, ""))}</p>`;
  }
  return inList ? html + "</ul>" : html;
}

function summarizeArgs(args) {
  return Object.entries(args || {})
    .map(([key, value]) => `${key} ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(" · ");
}

const gigabytes = (value) => `${Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 })} GB`;

// ---------------------------------------------------------------- header

async function refreshStatus() {
  app.status = await api("/api/status");
  renderHeader();
  renderComposer();
  return app.status;
}

function renderHeader() {
  const status = app.status;
  const model = status?.model;
  $("#model-name").textContent = model?.tag || "Choose a model";
  $("#model-dot").className = `dot ${status?.ready ? "ready" : status ? "problem" : ""}`;
  $("#model-button").title = status?.ready
    ? `Running on ${status.hardware.cpu}. Click to change the model.`
    : "Set up a model";
}

// ---------------------------------------------------------------- stage

function renderStage() {
  const session = app.session;
  $("#dropzone").hidden = Boolean(session);
  $("#viewer").hidden = !session;
  if (!session) return;

  const shown = app.comparing ? session.versions[0].id : session.current;
  const picture = $("#picture");
  const source = imageUrl(shown, 2048);
  if (picture.getAttribute("src") !== source) picture.src = source;
  $("#compare-badge").hidden = !app.comparing;
  $("#canvas").classList.toggle("working", app.busy);

  const idle = !app.busy;
  $("#undo-button").disabled = !(idle && session.can_undo);
  $("#redo-button").disabled = !(idle && session.can_redo);
  $("#open-button").disabled = !idle;
  $("#compare-button").disabled = session.cursor === 0;
  $("#compare-button").classList.toggle("active", app.comparing);

  const format = session.output.format.toUpperCase();
  const download = $("#download-button");
  download.textContent = `Download ${format}`;
  download.href = `/api/session/${session.id}/download`;
  download.setAttribute("aria-disabled", String(!idle));

  const edits = session.cursor === 1 ? "1 edit" : `${session.cursor} edits`;
  const transparent = session.transparent ? " · transparent" : "";
  $("#meta").textContent =
    `${session.filename} · ${session.width} × ${session.height} px${transparent} · ${edits}`;

  const list = $("#versions");
  list.replaceChildren(
    ...session.versions.map((version, index) => {
      const label = version.detail ? `${version.label}: ${version.detail}` : version.label;
      return h(
        "li",
        {},
        h(
          "button",
          {
            type: "button",
            class: `version ${index > session.cursor ? "undone" : ""}`,
            title: `${label} (${version.width} × ${version.height} px)`,
            "aria-current": String(index === session.cursor),
            disabled: !idle,
            onclick: () => gotoVersion(version.id),
          },
          h("img", { src: imageUrl(version.id, 192), alt: "", loading: "lazy" }),
          h("span", {}, version.label),
        ),
      );
    }),
  );
  list.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

async function sessionAction(path, json) {
  if (!app.session || app.busy) return;
  try {
    const data = await api(`/api/session/${app.session.id}/${path}`, { method: "POST", json });
    app.session = data.state;
    renderStage();
  } catch (error) {
    toast(error.message, "error");
  }
}

const gotoVersion = (version) => sessionAction("goto", { version });

function setComparing(on) {
  if (app.comparing === on || !app.session || (on && app.session.cursor === 0)) return;
  app.comparing = on;
  renderStage();
}

async function openFile(file) {
  if (!file) return;
  if (app.busy) return toast("Wait for the current task to finish, or stop it first.");
  const previous = app.session?.id;
  const form = new FormData();
  form.append("file", file);
  $("#dropzone").classList.add("loading");
  try {
    const data = await api("/api/session", { body: form });
    app.session = data.state;
    app.transcript = [];
    app.comparing = false;
    sessionStorage.setItem("pictor.session", app.session.id);
    if (previous) api(`/api/session/${previous}`, { method: "DELETE" }).catch(() => {});
    renderStage();
    renderChat();
    renderComposer();
    $("#prompt").focus();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("#dropzone").classList.remove("loading");
  }
}

async function restoreSession() {
  const id = sessionStorage.getItem("pictor.session");
  if (!id) return;
  try {
    const data = await api(`/api/session/${id}`);
    app.session = data.state;
    app.transcript = data.transcript;
    if (data.busy) waitForAgent(id);
  } catch {
    sessionStorage.removeItem("pictor.session");
  }
}

/** After a reload in the middle of a run, follow along until the agent is done. */
async function waitForAgent(id) {
  app.busy = true;
  renderComposer();
  while (app.session?.id === id) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    try {
      const data = await api(`/api/session/${id}`);
      app.session = data.state;
      app.transcript = data.transcript;
      app.busy = data.busy;
      renderStage();
      renderChat();
      if (!data.busy) break;
    } catch {
      break;
    }
  }
  app.busy = false;
  renderStage();
  renderComposer();
}

// ---------------------------------------------------------------- conversation

/** Build the DOM for one part of an assistant message. `update` re-reads the part. */
function partView(part) {
  if (part.type === "text") {
    const element = h("div", { class: "part text" });
    const update = () => (element.innerHTML = markdown(part.text));
    update();
    return { element, update };
  }
  if (part.type === "thinking") {
    const summary = h("summary");
    const body = h("div", { class: "thinking-body" });
    const element = h("details", { class: "part thinking" }, summary, body);
    const update = () => {
      summary.textContent = part.live ? "Thinking…" : "Thought about it";
      body.textContent = part.text.trim();
      if (part.live) body.scrollTop = body.scrollHeight;
    };
    update();
    return { element, update };
  }
  if (part.type === "tool") {
    const mark = h("span", { class: "tool-mark" });
    const summary = h("span", { class: "tool-summary" });
    const thumb = h("img", { class: "tool-thumb", alt: "", hidden: true });
    const body = h("div", { class: "tool-body", hidden: true });
    const head = h(
      "button",
      { type: "button", class: "tool-head", "aria-expanded": "false" },
      mark,
      h("span", { class: "tool-title" }, part.label || part.name),
      summary,
      thumb,
    );
    head.addEventListener("click", () => {
      body.hidden = !body.hidden;
      head.setAttribute("aria-expanded", String(!body.hidden));
    });
    const element = h("div", { class: "part tool" }, head, body);
    const update = () => {
      element.dataset.state = part.state;
      mark.textContent = part.state === "ok" ? "✓" : part.state === "failed" ? "!" : "";
      summary.textContent = part.status || summarizeArgs(part.args);
      if (part.version && app.session) {
        thumb.src = imageUrl(part.version.id, 192);
        thumb.hidden = false;
      }
      const details = [h("pre", {}, `${part.name}(${JSON.stringify(part.args || {}, null, 1)})`)];
      if (part.result) details.push(h("p", {}, part.result));
      body.replaceChildren(...details);
    };
    update();
    return { element, update };
  }
  if (part.type === "error") {
    return { element: h("div", { class: "part error" }, part.text), update() {} };
  }
  if (part.type === "stopped") {
    return { element: h("div", { class: "part stopped" }, "Stopped."), update() {} };
  }
  const element = h("div", { class: "part pending" }, part.text || "Working…");
  return { element, update: () => (element.textContent = part.text || "Working…") };
}

function messageView(message) {
  if (message.role === "user") {
    return { element: h("div", { class: "message user" }, message.text), views: new Map() };
  }
  const element = h("div", { class: "message assistant" });
  const views = new Map();
  for (const part of message.parts) {
    const view = partView(part);
    views.set(part, view);
    element.append(view.element);
  }
  return { element, views };
}

function scrollToEnd() {
  const messages = $("#messages");
  messages.scrollTop = messages.scrollHeight;
}

function renderChat() {
  const messages = $("#messages");
  if (!app.session) {
    messages.replaceChildren(
      h(
        "div",
        { class: "empty" },
        h("h2", {}, "Tell a local model what to do with a picture"),
        h("p", {}, "Choose a picture on the left to begin. Nothing leaves this computer."),
      ),
    );
    return;
  }
  if (!app.transcript.length) {
    messages.replaceChildren(
      h(
        "div",
        { class: "empty" },
        h("h2", {}, "What should be done with it?"),
        h("p", {}, "Describe the result you want, or start with one of these."),
        h(
          "div",
          { class: "suggestions" },
          SUGGESTIONS.map((text) =>
            h("button", { type: "button", class: "btn", onclick: () => send(text) }, text),
          ),
        ),
      ),
    );
    return;
  }
  messages.replaceChildren(...app.transcript.map((message) => messageView(message).element));
  scrollToEnd();
}

function renderComposer() {
  const prompt = $("#prompt");
  const button = $("#send-button");
  const ready = Boolean(app.status?.ready);
  const usable = ready && Boolean(app.session);
  prompt.disabled = !usable || app.busy;
  prompt.placeholder = !ready
    ? "Set up a model first"
    : !app.session
      ? "Open a picture first"
      : "Describe what to do with the picture…";
  button.textContent = app.busy ? "Stop" : "Send";
  button.classList.toggle("stop", app.busy);
  button.disabled = app.busy ? !app.controller : !usable;
}

function resizePrompt() {
  const prompt = $("#prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight + 2, 160)}px`;
}

async function send(text) {
  text = text.trim();
  if (!text || app.busy || !app.session || !app.status?.ready) return;

  const user = { role: "user", text };
  const reply = { role: "assistant", parts: [] };
  if (!app.transcript.length) $("#messages").replaceChildren();
  app.transcript.push(user, reply);
  const messages = $("#messages");
  messages.append(messageView(user).element);
  const view = messageView(reply);
  messages.append(view.element);

  // A placeholder until the first real piece of output arrives.
  let pending = { type: "pending", text: "Working…" };
  let pendingView = partView(pending);
  view.element.append(pendingView.element);
  const clearPending = () => {
    pendingView?.element.remove();
    pending = pendingView = null;
  };

  const addPart = (part) => {
    clearPending();
    const last = reply.parts.at(-1);
    if (last?.type === "thinking" && last.live) {
      last.live = false;
      view.views.get(last).update();
    }
    reply.parts.push(part);
    const created = partView(part);
    view.views.set(part, created);
    view.element.append(created.element);
    return part;
  };
  const streamInto = (type, delta) => {
    let part = reply.parts.at(-1);
    if (part?.type !== type) part = addPart({ type, text: "", live: type === "thinking" });
    part.text += delta;
    view.views.get(part).update();
  };
  const toolPart = (id) => reply.parts.find((part) => part.type === "tool" && part.id === id);

  app.busy = true;
  app.controller = new AbortController();
  $("#prompt").value = "";
  resizePrompt();
  renderComposer();
  renderStage();
  scrollToEnd();

  try {
    const response = await api(`/api/session/${app.session.id}/chat`, {
      json: { message: text },
      signal: app.controller.signal,
      raw: true,
    });
    for await (const event of ndjson(response)) {
      const pinned = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
      if (event.type === "status" && pending) {
        pending.text = event.text;
        pendingView.update();
      } else if (event.type === "thinking" || event.type === "text") {
        streamInto(event.type, event.delta);
      } else if (event.type === "tool_start") {
        // The event's own `type` ("tool_start") must not replace the part type.
        addPart({ ...event, type: "tool", state: "running" });
      } else if (event.type === "tool_status") {
        const part = toolPart(event.id);
        if (part) {
          part.status = event.text;
          view.views.get(part).update();
        }
      } else if (event.type === "tool_end") {
        const part = toolPart(event.id);
        if (part) {
          Object.assign(part, {
            state: event.ok ? "ok" : "failed",
            result: event.result,
            version: event.version,
            status: "",
          });
          view.views.get(part).update();
        }
      } else if (event.type === "state") {
        app.session = event.state;
        renderStage();
      } else if (event.type === "error") {
        addPart({ type: "error", text: event.message });
      }
      if (pinned) scrollToEnd();
    }
  } catch (error) {
    if (error.name === "AbortError") addPart({ type: "stopped" });
    else addPart({ type: "error", text: error.message });
  } finally {
    clearPending();
    for (const part of reply.parts) {
      if (part.type === "thinking") part.live = false;
      if (part.type === "tool" && part.state === "running") {
        part.state = "failed";
        part.result = "Stopped.";
      }
      view.views.get(part)?.update();
    }
    app.busy = false;
    app.controller = null;
    try {
      // The server is the source of truth for versions, especially after a stop.
      app.session = (await api(`/api/session/${app.session.id}`)).state;
    } catch {
      // Keep the state from the stream.
    }
    renderStage();
    renderComposer();
    scrollToEnd();
    $("#prompt").focus();
  }
}

// ---------------------------------------------------------------- model setup

function fitBadge(option) {
  if (!option.usable) return h("span", { class: "badge unusable" }, "No tool calling");
  const text = { fits: "Fits", tight: "Tight fit", too_big: "Too large" }[option.fit];
  return h("span", { class: `badge ${option.fit}` }, text);
}

function selectedOption() {
  const { models, choice, custom } = app.setup;
  const tag = custom.trim() || choice;
  return { tag, option: models?.options.find((option) => option.tag === tag) };
}

function renderSetup() {
  const body = $("#setup-body");
  const status = app.status;
  const setup = app.setup;
  const firstRun = !status?.ready;
  const hardware = status?.hardware;

  const head = h(
    "div",
    { class: "setup-head" },
    h("h1", { id: "setup-title" }, firstRun ? "Welcome to Pictor" : "Model"),
    h(
      "p",
      {},
      "Pictor runs an AI model on this computer to edit your pictures. ",
      "Nothing is sent anywhere.",
    ),
  );

  const scroll = h("div", { class: "setup-scroll" });
  const foot = h("div", { class: "setup-foot" });

  if (hardware) {
    const gpus = hardware.gpus
      .filter((gpu) => gpu.discrete)
      .map((gpu) => h("span", { class: "fact" }, gpu.vram_gb ? `${gpu.name} · ${gigabytes(gpu.vram_gb)}` : gpu.name));
    scroll.append(
      h(
        "div",
        { class: "panel" },
        h("h2", {}, "This computer"),
        h(
          "div",
          { class: "facts" },
          h("span", { class: "fact" }, hardware.cpu),
          h("span", { class: "fact" }, `${gigabytes(hardware.ram_gb)} memory`),
          gpus,
          h("span", { class: "fact" }, `${hardware.os} ${hardware.os_version}`),
        ),
        h("p", { class: "muted" }, hardware.budget_note),
      ),
    );
  }

  if (!status?.ollama.running) {
    const installed = status?.ollama.installed;
    scroll.append(
      h(
        "div",
        { class: "panel problem" },
        h("h2", {}, installed ? "Ollama is not running" : "Ollama is needed"),
        h(
          "p",
          {},
          "Pictor uses Ollama, a free open-source program, to run the model on your machine.",
        ),
        installed
          ? h("p", {}, "It is installed but not answering. Start the Ollama app, or run ", h("code", {}, "ollama serve"), " in a terminal.")
          : h(
              "ol",
              {},
              h("li", {}, "Download and install it from ", h("a", { href: status?.ollama.download_url || "https://ollama.com/download", target: "_blank", rel: "noreferrer" }, "ollama.com/download"), "."),
              h("li", {}, "Start it. This page continues by itself once it is running."),
            ),
      ),
    );
    foot.append(
      h("span", { class: "grow" }, "Checking every few seconds…"),
      h("button", { type: "button", class: "btn primary", onclick: loadSetup }, "Check again"),
    );
    body.replaceChildren(head, scroll, foot);
    return;
  }

  if (!setup.models) {
    scroll.append(h("div", { class: "part pending" }, setup.error || "Looking at the available models…"));
    body.replaceChildren(head, scroll, foot);
    return;
  }

  const pulling = Boolean(setup.pull);
  const modelRow = (option) => {
    const input = h("input", {
      type: "radio",
      name: "model",
      value: option.tag,
      checked: !setup.custom.trim() && setup.choice === option.tag,
      disabled: !option.usable || pulling,
      onchange: () => {
        setup.choice = option.tag;
        setup.custom = "";
        renderSetup();
      },
    });
    const facts = [`needs about ${gigabytes(option.need_gb)} of memory`];
    if (!option.installed) facts.unshift(`${gigabytes(option.size_gb)} download`);
    if (option.reason || option.note) facts.push(option.reason || option.note);
    return h(
      "label",
      { class: `model ${option.usable ? "" : "unusable"}` },
      input,
      h(
        "span",
        { class: "model-name" },
        option.name,
        option.installed && h("span", { class: "badge" }, "Installed"),
        !option.vision && option.usable && h("span", { class: "badge tight" }, "Cannot see pictures"),
      ),
      fitBadge(option),
      h(
        "span",
        { class: "model-sub" },
        // Models from outside the catalog are already named by their tag.
        option.name !== option.tag && h("code", {}, option.tag),
        option.name !== option.tag && " · ",
        facts.join(" · "),
      ),
    );
  };
  const recommended = setup.models.options.find((option) => option.recommended);
  const others = setup.models.options.filter((option) => option !== recommended);
  const list = h(
    "div",
    { class: "model-groups", role: "radiogroup", "aria-label": "Model" },
    recommended &&
      h(
        "div",
        { class: "model-list" },
        h("h2", {}, "Recommended for this computer"),
        modelRow(recommended),
      ),
    h(
      "div",
      { class: "model-list" },
      h("h2", {}, recommended ? "Other models" : "Models"),
      h(
        "p",
        { class: "muted" },
        "Bigger models follow instructions better. One that does not fit in memory runs slowly, or not at all.",
      ),
      others.map(modelRow),
    ),
  );

  const custom = h("input", {
    id: "custom-model",
    type: "text",
    value: setup.custom,
    placeholder: "e.g. qwen3.5:9b",
    autocomplete: "off",
    spellcheck: "false",
    disabled: pulling,
    oninput: (event) => {
      const had = Boolean(setup.custom.trim());
      setup.custom = event.target.value;
      // Re-render only when the radio selection has to change, to keep typing smooth.
      if (had !== Boolean(setup.custom.trim())) {
        renderSetup();
        const field = $("#custom-model");
        field.focus();
        field.setSelectionRange(field.value.length, field.value.length);
      } else {
        renderSetupFoot(foot);
      }
    },
  });
  const think = h(
    "select",
    { id: "think-mode", disabled: pulling, onchange: (event) => saveThink(event.target.value) },
    [
      ["auto", "Automatic: brief reasoning when the model supports it"],
      ["off", "Off: fastest"],
      ["default", "Model default: most thorough, slowest"],
    ].map(([value, label]) => h("option", { value, selected: status.think === value }, label)),
  );

  scroll.append(
    list,
    h(
      "details",
      { class: "advanced", open: Boolean(setup.custom.trim()) },
      h("summary", {}, "More options"),
      h(
        "div",
        {},
        h(
          "div",
          { class: "field" },
          h("label", { for: "custom-model" }, "Another Ollama model"),
          custom,
          h("span", { class: "muted" }, "Any tag from ollama.com/library. It must support tool calling; vision is strongly recommended."),
        ),
        h("div", { class: "field" }, h("label", { for: "think-mode" }, "Reasoning before acting"), think),
        h("span", { class: "muted" }, `Model list last updated ${setup.models.catalog_updated}.`),
      ),
    ),
  );

  renderSetupFoot(foot);
  body.replaceChildren(head, scroll, foot);
}

function renderSetupFoot(foot) {
  const setup = app.setup;
  const firstRun = !app.status?.ready;
  const { tag, option } = selectedOption();
  const children = [];

  if (setup.pull) {
    const { percent, completed, total, status } = setup.pull;
    const known = typeof percent === "number";
    const detail = known
      ? `${gigabytes(completed / 1e9)} of ${gigabytes(total / 1e9)} (${Math.round(percent)}%)`
      : status || "Starting…";
    children.push(
      h(
        "div",
        { class: "grow" },
        h("div", {}, `Downloading ${tag} · ${detail}`),
        h("div", { class: `progress ${known ? "" : "indeterminate"}` }, h("div", { style: known ? `width:${percent}%` : "" })),
      ),
      h("button", { type: "button", class: "btn", onclick: cancelPull }, "Cancel"),
    );
  } else {
    let note = setup.error;
    let noteClass = setup.error ? "grow warning" : "grow";
    if (!note && option && option.usable && option.fit !== "fits") {
      note = `Needs about ${gigabytes(option.need_gb)} but about ${gigabytes(setup.models.budget_gb)} is available, so it will be slow.`;
      noteClass = "grow warning";
    }
    children.push(h("span", { class: noteClass }, note || ""));
    if (!firstRun) {
      children.push(h("button", { type: "button", class: "btn", onclick: closeSetup }, "Close"));
    }
    const installed = Boolean(option?.installed);
    const label = !tag
      ? "Choose a model"
      : installed
        ? tag === app.status?.model?.tag && !firstRun
          ? "In use"
          : "Use this model"
        : option
          ? `Download (${gigabytes(option.size_gb)}) and use`
          : "Download and use";
    const disabled = !tag || (option && !option.usable) || label === "In use";
    children.push(h("button", { type: "button", class: "btn primary", disabled, onclick: confirmModel }, label));
  }
  foot.replaceChildren(...children);
}

async function loadSetup() {
  const setup = app.setup;
  clearTimeout(setup.timer);
  try {
    await refreshStatus();
    if (!app.status.ollama.running) {
      setup.models = null;
      renderSetup();
      // Keep looking until Ollama appears, as long as the dialog is open.
      if ($("#setup").open) setup.timer = setTimeout(loadSetup, 3000);
      return;
    }
    setup.models = await api("/api/models");
    const tags = setup.models.options.filter((option) => option.usable).map((option) => option.tag);
    if (!tags.includes(setup.choice)) {
      setup.choice = tags.includes(setup.models.selected)
        ? setup.models.selected
        : (setup.models.options.find((option) => option.recommended)?.tag ?? tags[0] ?? null);
    }
    setup.error = "";
  } catch (error) {
    setup.error = error.message;
  }
  renderSetup();
}

function openSetup() {
  const dialog = $("#setup");
  app.setup.error = "";
  renderSetup();
  if (!dialog.open) dialog.showModal();
  loadSetup();
}

function closeSetup() {
  if (!app.status?.ready || app.setup.pull) return;
  clearTimeout(app.setup.timer);
  $("#setup").close();
}

async function saveThink(value) {
  try {
    app.status = await api("/api/settings", { json: { think: value } });
  } catch (error) {
    toast(error.message, "error");
  }
}

function cancelPull() {
  app.setup.pull?.controller.abort();
}

async function confirmModel() {
  const setup = app.setup;
  const { tag, option } = selectedOption();
  if (!tag || setup.pull) return;
  setup.error = "";
  try {
    if (!option?.installed) {
      const controller = new AbortController();
      setup.pull = { controller, status: "Starting…" };
      renderSetup();
      const response = await api("/api/models/pull", {
        json: { model: tag },
        signal: controller.signal,
        raw: true,
      });
      for await (const event of ndjson(response)) {
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "progress") {
          Object.assign(setup.pull, event);
          renderSetupFoot($(".setup-foot"));
        }
      }
    }
    setup.pull = null;
    app.status = await api("/api/models/select", { json: { model: tag } });
    setup.custom = "";
    setup.choice = tag;
    renderHeader();
    renderComposer();
    clearTimeout(setup.timer);
    $("#setup").close();
    toast(`${tag} is ready.`);
  } catch (error) {
    setup.pull = null;
    setup.error = error.name === "AbortError" ? "Download cancelled. It resumes where it stopped next time." : error.message;
    renderSetup();
  }
}

// ---------------------------------------------------------------- wiring

function bind() {
  const fileInput = $("#file-input");
  const pick = () => fileInput.click();
  $("#dropzone").addEventListener("click", pick);
  $("#open-button").addEventListener("click", pick);
  fileInput.addEventListener("change", () => {
    openFile(fileInput.files[0]);
    fileInput.value = "";
  });

  const stage = $("#stage");
  let depth = 0;
  stage.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    stage.classList.add("dragging");
  });
  stage.addEventListener("dragover", (event) => event.preventDefault());
  stage.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) stage.classList.remove("dragging");
  });
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    stage.classList.remove("dragging");
    openFile(event.dataTransfer?.files?.[0]);
  });
  // Dropping a file outside the stage must not navigate away from the app.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  document.addEventListener("paste", (event) => {
    const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/"));
    if (file) {
      event.preventDefault();
      openFile(file);
    }
  });

  $("#undo-button").addEventListener("click", () => sessionAction("undo"));
  $("#redo-button").addEventListener("click", () => sessionAction("redo"));
  $("#download-button").addEventListener("click", (event) => {
    if (app.busy) event.preventDefault();
  });

  const compare = $("#compare-button");
  compare.addEventListener("pointerdown", () => setComparing(true));
  for (const name of ["pointerup", "pointerleave", "pointercancel", "blur"]) {
    compare.addEventListener(name, () => setComparing(false));
  }
  compare.addEventListener("keydown", (event) => {
    if (event.key === " " || event.key === "Enter") setComparing(true);
  });
  compare.addEventListener("keyup", () => setComparing(false));

  const prompt = $("#prompt");
  prompt.addEventListener("input", resizePrompt);
  prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(prompt.value);
    }
  });
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (app.busy) app.controller?.abort();
    else send(prompt.value);
  });

  document.addEventListener("keydown", (event) => {
    const typing = ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement?.tagName);
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z" && !typing) {
      event.preventDefault();
      sessionAction(event.shiftKey ? "redo" : "undo");
    }
  });

  $("#model-button").addEventListener("click", openSetup);
  const dialog = $("#setup");
  // Esc closes the dialog only once a working model exists.
  dialog.addEventListener("cancel", (event) => {
    if (!app.status?.ready || app.setup.pull) event.preventDefault();
    else clearTimeout(app.setup.timer);
  });
}

async function boot() {
  bind();
  renderChat();
  try {
    await refreshStatus();
  } catch (error) {
    toast(`The Pictor server is not answering: ${error.message}`, "error");
    return;
  }
  await restoreSession();
  renderStage();
  renderChat();
  renderComposer();
  if (!app.status.ready) openSetup();
}

boot();
