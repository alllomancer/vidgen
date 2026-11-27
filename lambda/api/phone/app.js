// vidgen phone page. Talks only to this API, signing every call with a key that never leaves this browser.
// Everything shown from the server goes in as text (textContent / text nodes), never parsed as HTML. docs/specs/2026-11-27-phone-ui.md
import { COMMENT, Offline, Unauthorized, loadOrCreateKey, publicKeyLine, signedFetch } from "/m/sign.js";

const REFRESH_MS = 10000;
const NAME_KEY = "vidgen.keyName";
let key = null;
let timer = null;
let session = 0; // bumped by start(): answers to requests from an earlier session are ignored
let listGen = 0; // bumped by each history load: only the newest one draws
let starting = false;
const STATUSES = ["queued", "running", "done", "failed", "cancelled"];
const pace = new Map(); // job id -> {phase, n0, t0, n, t}: progress seen on this page, for "time left"

// ---------------------------------------------------------------- small helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}

const $ = (id) => document.getElementById(id);

function show(view) {
  const changed = $(view).hidden;
  for (const s of document.querySelectorAll("main > section")) s.hidden = s.id !== view;
  for (const b of document.querySelectorAll("nav button")) {
    if (b.dataset.view === view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  $("nav").hidden = view === "pair" || view === "boot";
  const heading = $(view).querySelector("h2");
  if (changed && heading) heading.focus(); // screen readers hear the new view; focus isn't left on a hidden button
}

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

function storedName() {
  try { return localStorage.getItem(NAME_KEY) || ""; } catch (e) { return ""; }
}

function storeName(name) {
  try { localStorage.setItem(NAME_KEY, name); } catch (e) { /* private mode: asked again next time */ }
}

function roughly(s) {
  if (s < 90) return `${Math.max(1, Math.round(s))} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied");
  } catch (e) {
    toast("Copying isn't allowed here: select the text and copy it");
  }
}

async function call(method, path, body) {
  const mine = session;
  try {
    return await signedFetch(key, method, path, body);
  } catch (e) {
    if (e instanceof Unauthorized && mine === session) pairScreen(e);
    throw e;
  }
}

/** Run an action from a button: disabled while it runs, so a double tap doesn't send it twice. */
async function busy(button, fn) {
  button.disabled = true;
  try {
    await fn();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message);
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- pairing

function pairScreen(err) {
  stopRefresh();
  show("pair");
  const name = $("pair-name");
  if (!name.value) name.value = storedName();
  $("pair-reason").textContent = err.clock ? "This phone's clock is off, so the server can't trust its requests. "
    + "Turn on automatic date and time in Android settings, then tap Try again." : "This phone isn't paired yet.";
  $("pair-steps").hidden = !!err.clock;
  renderLine();
}

function renderLine() {
  const name = $("pair-name").value.trim();
  const ok = !name || COMMENT.test(name);
  $("pair-name-error").hidden = ok;
  $("pair-line").value = ok && name ? publicKeyLine(key.pubRaw, name) : "";
  $("pair-copy").disabled = !(ok && name);
}

function setupPairing() {
  $("pair-name").addEventListener("input", renderLine);
  $("pair-copy").addEventListener("click", () => {
    storeName($("pair-name").value.trim());
    copyText($("pair-line").value);
  });
  $("pair-retry").addEventListener("click", () => {
    storeName($("pair-name").value.trim());
    start();
  });
  $("boot-retry").addEventListener("click", () => start());
}

// ---------------------------------------------------------------- history

function finishTime(j, n, of) {
  const phase = j.progress.replace(/\d+/g, "#");
  const now = Date.now() / 1000;
  let p = pace.get(j.job_id);
  if (!p || p.phase !== phase || n < p.n) {
    p = { phase, n0: n, t0: now, n, t: now };
    pace.set(j.job_id, p);
  }
  if (n > p.n) { p.n = n; p.t = now; }
  return p.n > p.n0 && n < of ? p.t + ((of - n) * (p.t - p.t0)) / (p.n - p.n0) : null;
}

function statusPart(j) {
  const status = STATUSES.includes(j.status) ? j.status : "other";
  const parts = [el("span", { class: `pill ${status}` }, String(j.status || "unknown"))];
  if (j.status === "queued" && j.created_at) {
    parts.push(el("div", { class: "sub" }, `waiting ${roughly(Date.now() / 1000 - j.created_at)}`));
  }
  if (j.status === "running") {
    const text = String(j.progress || "");
    const m = /(\d+)\s*\/\s*(\d+)/.exec(text);
    const n = m && Number(m[1]);
    const of = m && Number(m[2]);
    if (text) parts.push(el("div", { class: "sub" }, text));
    if (m && of > 0 && n <= of) {
      const pct = Math.round((100 * n) / of);
      const bar = el("div", { class: "bar", role: "progressbar", "aria-label": text, "aria-valuenow": pct,
                              "aria-valuemin": 0, "aria-valuemax": 100 }, el("span"));
      bar.firstChild.style.width = `${pct}%`;
      parts.push(bar);
      const eta = /\bleft\b/.test(text) ? null : finishTime(j, n, of);
      const left = eta === null ? null : eta - Date.now() / 1000;
      const ran = j.started_at ? `running ${roughly(Date.now() / 1000 - j.started_at)}` : "";
      const leftText = left === null ? "" : left < 60 ? "under a minute left" : `about ${roughly(left)} left`;
      if (ran || leftText) parts.push(el("div", { class: "sub" }, [ran, leftText].filter(Boolean).join(" · ")));
    }
  }
  if (j.error) parts.push(el("div", { class: "sub err" }, String(j.error)));
  return parts;
}

const COPYABLE = ["video", "photos", "avatar"];

function jobCard(j) {
  const actions = [];
  if (j.status === "queued" || j.status === "running") {
    actions.push(el("button", { class: "small", onclick: (e) => cancel(j, e.currentTarget) }, "Cancel"));
  } else if (["video", "photos", "story", "write", "avatar"].includes(j.type)) {
    actions.push(el("button", { class: "small", onclick: (e) => rerun(j, e.currentTarget) }, "Re-run"));
  }
  if (j.summary && COPYABLE.includes(j.type)) {
    actions.push(el("button", { class: "small", onclick: () => copyText(j.summary) }, "Copy prompt"));
  }
  const when = j.created_at ? new Date(j.created_at * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : "";
  const source = j.meta && j.meta.source ? ` · from ${j.meta.source}` : "";
  return el("li", { class: "job" },
    el("div", { class: "row" }, el("strong", {}, j.type || "job"), el("span", { class: "muted" }, when + source)),
    el("div", { class: "summary" }, String(j.summary || "")),
    el("div", { class: "status" }, statusPart(j)),
    actions.length ? el("div", { class: "actions" }, actions) : null);
}

let lastLoaded = null;

async function refreshHistory() {
  const mine = ++listGen;
  const r = await call("GET", "/jobs");
  if (mine !== listGen) return; // a newer load started meanwhile; it draws
  const jobs = Array.isArray(r.jobs) ? r.jobs.filter((j) => j && typeof j === "object") : [];
  const ids = new Set(jobs.map((j) => j.job_id));
  for (const id of pace.keys()) if (!ids.has(id)) pace.delete(id);
  $("history-list").replaceChildren(...jobs.map(jobCard));
  $("history-empty").hidden = jobs.length > 0;
  lastLoaded = new Date();
  $("history-updated").textContent = `Updated ${lastLoaded.toLocaleTimeString([], { timeStyle: "short" })}`;
  $("history-stale").hidden = true;
}

/** A background refresh failed: keep the list, say it may be out of date. */
function staleNote(e) {
  if (e instanceof Unauthorized) return;
  const since = lastLoaded ? `; showing the list from ${lastLoaded.toLocaleTimeString([], { timeStyle: "short" })}` : "";
  $("history-stale").textContent = `${e.message}${since}`;
  $("history-stale").hidden = false;
}

function cancel(j, button) {
  return busy(button, async () => {
    const r = await call("POST", `/jobs/${encodeURIComponent(j.job_id)}/cancel`);
    toast(r.status === "cancelled" ? "Cancelled" : "Stopping within ~15 s");
    refreshHistory().catch(staleNote);
  });
}

function rerun(j, button) {
  return busy(button, async () => {
    await call("POST", `/jobs/${encodeURIComponent(j.job_id)}/rerun`);
    toast("Queued again");
    refreshHistory().catch(staleNote);
  });
}

function stopRefresh() {
  clearInterval(timer);
  timer = null;
}

function startRefresh() {
  stopRefresh();
  timer = setInterval(() => {
    if (document.visibilityState === "visible" && !$("history").hidden) refreshHistory().catch(staleNote);
  }, REFRESH_MS);
}

// ---------------------------------------------------------------- start

async function persistHint() {
  let persisted = false;
  try {
    persisted = (await navigator.storage?.persisted?.()) || (await navigator.storage?.persist?.()) || false;
  } catch (e) { /* unknown: show the hint */ }
  $("install-hint").hidden = persisted;
}

async function ed25519Supported() {
  try {
    await crypto.subtle.generateKey({ name: "Ed25519" }, false, ["sign"]);
    return true;
  } catch (e) {
    return false;
  }
}

async function start() {
  if (starting) return;
  starting = true;
  session += 1;
  stopRefresh();
  $("pair-retry").disabled = true;
  $("boot-retry").hidden = true;
  show("boot");
  $("boot-text").textContent = "Loading…";
  try {
    await refreshHistory();
    show("history");
    startRefresh();
  } catch (e) {
    if (!(e instanceof Unauthorized)) {
      $("boot-text").textContent = e instanceof Offline ? e.message : `Couldn't load the history: ${e.message}`;
      $("boot-retry").hidden = false;
    }
  } finally {
    starting = false;
    $("pair-retry").disabled = false;
  }
}

async function main() {
  if (!window.isSecureContext || !(await ed25519Supported())) {
    $("boot-text").textContent =
      "This browser can't make the signing key this page needs. Update Chrome (version 137 or later).";
    return;
  }
  try {
    key = await loadOrCreateKey();
  } catch (e) {
    $("boot-text").textContent = `This page can't store its key (${e.message}). Close other vidgen tabs, make sure `
      + "Chrome isn't in Incognito mode and site data is allowed, then reload.";
    return;
  }
  setupPairing();
  for (const b of document.querySelectorAll("nav button")) {
    b.addEventListener("click", () => {
      show(b.dataset.view);
      if (b.dataset.view === "history") refreshHistory().catch(staleNote);
    });
  }
  $("history-refresh").addEventListener("click", (e) => busy(e.currentTarget, refreshHistory));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && !$("history").hidden) refreshHistory().catch(staleNote);
  });
  persistHint();
  start();
}

main().catch((e) => {
  $("boot-text").textContent = `Something went wrong: ${e.message}`;
});
