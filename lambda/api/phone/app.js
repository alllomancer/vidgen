// vidgen phone page. Talks only to this API, signing every call with a key that never leaves this browser.
// Everything shown from the server goes in as text (textContent / text nodes), never parsed as HTML. docs/specs/2026-11-27-phone-ui.md
import {
  COMMENT, Offline, Unauthorized, loadOrCreateKey, publicKeyLine, putUpload, requestPairing, signedFetch,
} from "/m/sign.js";

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

let pairPoll = null;

function stopPairing() {
  clearTimeout(pairPoll);
  pairPoll = null;
}

/** Pair this phone: get a code, show it, and check every 5 s whether the Mac approved it. */
async function startPairing(button) {
  stopPairing();
  $("pair-error").textContent = "";
  button.disabled = true;
  let p;
  const asked = session;
  try {
    p = await requestPairing(key);
  } catch (e) {
    $("pair-error").textContent = e.message;
    button.disabled = false;
    return;
  }
  button.disabled = false;
  if (asked !== session) return; // the page moved on meanwhile
  $("pair-code").textContent = `${p.code.slice(0, 3)} ${p.code.slice(3)}`;
  $("pair-fingerprint").textContent = String(p.fingerprint || "");
  $("pair-wait").hidden = false;
  button.textContent = "Get a new code";
  const mine = session;
  const expires = Number(p.expires_at) * 1000 || Date.now() + 600000;
  const check = async () => {
    if (mine !== session) return;
    if (Date.now() > expires) {
      $("pair-wait").hidden = true;
      $("pair-error").textContent = "The code expired. Tap Get a new code.";
      return;
    }
    try {
      await signedFetch(key, "GET", "/status");
      stopPairing();
      toast("Paired");
      start();
      return;
    } catch (e) {
      // not approved yet: say nothing; the clock off: say so (approving won't help until it's fixed)
      if (e instanceof Unauthorized) $("pair-error").textContent = e.clock ? e.message : "";
      else if (!(e instanceof Offline)) $("pair-error").textContent = e.message;
    }
    pairPoll = setTimeout(check, 5000);
  };
  pairPoll = setTimeout(check, 5000);
}

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
  $("pair-start").addEventListener("click", (e) => startPairing(e.currentTarget));
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

// ---------------------------------------------------------------- AI-look check (docs/specs/2026-11-27-realism-check.md)

const REMEMBER_CHECK = "vidgen.realismCheck";

function realismLevel(score) {
  return score <= 3 ? "low" : score <= 6 ? "mid" : "high";
}

function realismBadge(r) {
  if (!r || typeof r !== "object") return null;
  if (typeof r.score !== "number") return el("div", { class: "sub" }, `AI-look check: ${String(r.error || "no verdict")}`);
  const words = { low: "looks real", mid: "some giveaways", high: "looks AI-made" }[realismLevel(r.score)];
  return el("span", { class: `badge r-${realismLevel(r.score)}` }, `AI look ${Number(r.score)}/10 · ${words}`);
}

function showVerdict(job) {
  const box = $("job-verdict");
  const r = job.result && job.result.realism && typeof job.result.realism === "object" ? job.result.realism : null;
  if (!r) { box.hidden = true; box.replaceChildren(); return; }
  const parts = [el("h3", {}, typeof r.score === "number" ? `AI look ${Number(r.score)}/10` : "AI-look check")];
  if (r.error) parts.push(el("p", { class: "err" }, String(r.error)));
  const signs = Array.isArray(r.signs) ? r.signs.filter((x) => typeof x === "string") : [];
  if (signs.length) parts.push(el("h3", {}, "What gives it away"), el("ul", {}, signs.map((x) => el("li", {}, x))));
  const prompt = typeof r.prompt === "string" ? r.prompt : "";
  if (prompt) parts.push(el("h3", {}, "Suggested prompt"), el("p", { class: "suggest" }, prompt));
  const scenes = Array.isArray(r.scenes) ? r.scenes.filter((x) => typeof x === "string") : [];
  if (scenes.length) parts.push(el("h3", {}, "Suggested scenes"), el("ol", {}, scenes.map((x) => el("li", { class: "suggest" }, x))));
  parts.push(el("p", { class: "sub" }, String(r.note || "An AI critique of visible artifacts, not a detector.")));
  const actions = [];
  if (prompt || scenes.length) {
    actions.push(el("button", { class: "small primary", onclick: () => useSuggestion(job.type, r) }, "Use suggestion"));
  }
  if (prompt) actions.push(el("button", { class: "small", onclick: () => copyText(prompt) }, "Copy prompt"));
  if (actions.length) parts.push(el("div", { class: "actions" }, actions));
  box.replaceChildren(...parts);
  box.hidden = false;
}

/** Fill the Create form of the job's kind with the suggestion (not sent until you tap the button there). */
function useSuggestion(kind, r) {
  const form = document.querySelector(`form[data-kind="${kind}"]`);
  if (!form) return;
  leaveJob();
  show("create");
  showKind(kind);
  loadAvatars();
  const f = form.elements;
  if (kind === "video" && typeof r.prompt === "string" && r.prompt) f.prompt.value = r.prompt;
  if (kind === "photos" && Array.isArray(r.scenes) && r.scenes.length) {
    f.scenes.value = r.scenes.filter((x) => typeof x === "string").slice(0, 8).join("\n");
    f.count.value = ""; // one photo per suggested scene
  }
  toast("Suggested prompt loaded: check it, then render again");
}

function setupRealismToggle() {
  let remembered = false;
  try { remembered = localStorage.getItem(REMEMBER_CHECK) === "1"; } catch (e) { /* not remembered */ }
  const boxes = document.querySelectorAll('input[name="realism_check"][data-remember]');
  for (const b of boxes) {
    b.checked = remembered;
    b.addEventListener("change", () => {
      for (const other of boxes) other.checked = b.checked;
      try { localStorage.setItem(REMEMBER_CHECK, b.checked ? "1" : "0"); } catch (e) { /* not remembered */ }
    });
  }
}

function jobCard(j) {
  const actions = [];
  if (j.status === "queued" || j.status === "running") {
    actions.push(el("button", { class: "small", onclick: (e) => cancel(j, e.currentTarget) }, "Cancel"));
  } else if (["video", "photos", "story", "write", "avatar"].includes(j.type)) {
    actions.push(el("button", { class: "small", onclick: (e) => rerun(j, e.currentTarget) }, "Re-run"));
  }
  if (j.status === "done" && j.type !== "train") {
    actions.unshift(el("button", { class: "small", onclick: () => openJob(j.job_id) }, "Open"));
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
    realismBadge(j.realism),
    actions.length ? el("div", { class: "actions" }, actions) : null);
}

let lastLoaded = null;

/** The render box's state on top of History: off, starting or running, and how many jobs wait. */
async function refreshStatus() {
  let s;
  try {
    s = await call("GET", "/status");
  } catch (e) {
    return; // the history line already says when the server can't be reached
  }
  const state = String(s.instance || "unknown");
  const queued = Number(s.queued_jobs) || 0;
  const words = {
    stopped: "Render box is off; it starts when a job is queued",
    stopping: "Render box is stopping",
    pending: "Render box is starting (a few minutes)",
    running: `Render box is on${s.running_minutes ? ` (${roughly(s.running_minutes * 60)})` : ""}`,
  };
  const text = words[state] || `Render box: ${state}`;
  $("status-text").textContent = queued ? `${text} · ${queued} job${queued > 1 ? "s" : ""} waiting` : text;
  $("status-strip").dataset.state = ["running", "pending", "stopped", "stopping"].includes(state) ? state : "other";
  $("status-strip").hidden = false;
}

async function refreshHistory() {
  refreshStatus();
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

// ---------------------------------------------------------------- one job: results and Save

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;
const VIDEO_EXT = /\.(mp4|webm|mov)$/i;
let openJobId = null;

function fileExt(url) {
  try { return new URL(url).pathname; } catch (e) { return ""; }
}

/** Only links to this bucket's presigned objects are shown or followed. */
function bucketLink(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" && u.hostname.endsWith(".amazonaws.com") ? u.href : null;
  } catch (e) {
    return null;
  }
}

async function openJob(jobId) {
  openJobId = jobId;
  show("job");
  $("job-title").textContent = "Loading…";
  $("job-files").replaceChildren();
  $("job-note").textContent = "";
  let j;
  try {
    j = await call("GET", `/jobs/${encodeURIComponent(jobId)}`); // fresh links: they last an hour
  } catch (e) {
    if (openJobId === jobId && !(e instanceof Unauthorized)) $("job-title").textContent = e.message;
    return;
  }
  if (openJobId !== jobId) return;
  $("job-title").textContent = `${j.type || "job"} · ${j.status || ""}`;
  showVerdict(j);
  const downloads = j.downloads && typeof j.downloads === "object" ? j.downloads : {};
  const saves = j.saves && typeof j.saves === "object" ? j.saves : {};
  const names = Object.keys(downloads).sort((a, b) => (a === "video" ? -1 : b === "video" ? 1 : a.localeCompare(b)));
  const items = [];
  for (const name of names) {
    const view = bucketLink(downloads[name]);
    if (!view) continue;
    const path = fileExt(view);
    const save = bucketLink(saves[name]);
    let media = null;
    if (IMAGE_EXT.test(path)) media = el("img", { src: view, alt: `${name} from this job`, loading: "lazy" });
    else if (VIDEO_EXT.test(path)) media = el("video", { src: view, controls: true, playsinline: true, preload: "metadata" });
    items.push(el("li", { class: "file" },
      media,
      el("div", { class: "row" }, el("span", {}, name.replace(/_/g, " ")),
        save ? el("a", { class: "button small", href: save, rel: "noopener" }, "Save") : null)));
  }
  $("job-files").replaceChildren(...items);
  $("job-note").textContent = items.length ? "Save downloads the file; it shows up in Files → Downloads and the gallery."
                                           : "Nothing to show for this job.";
}

/** Leaving the Job screen: stop any video and forget the job. */
function leaveJob() {
  openJobId = null;
  $("job-files").replaceChildren();
  $("job-verdict").replaceChildren();
  $("job-verdict").hidden = true;
}

// ---------------------------------------------------------------- avatars

const MAX_AVATAR_IMAGES = 12;

function avatarCard(a, def) {
  const name = String(a.name);
  const states = [`${Number(a.images) || 0} photo${Number(a.images) === 1 ? "" : "s"}`];
  for (const [label, rec] of [["Krea LoRA", a.krea_lora], ["Wan LoRA", a.wan_lora]]) {
    if (rec && typeof rec.state === "string" && rec.state !== "none") states.push(`${label}: ${rec.state}`);
  }
  const input = el("input", { type: "file", accept: "image/*", multiple: true, hidden: true, tabindex: "-1" });
  const status = el("span", { class: "form-status", role: "status" });
  const add = el("button", { type: "button", class: "small", "aria-label": `Add photos to ${name}`,
                             onclick: () => input.click() }, "Add photos");
  input.addEventListener("change", () => addAvatarPhotos(name, input, add, status, Number(a.images) || 0));
  const actions = [add, input];
  if (name !== def) {
    actions.push(el("button", { class: "small", onclick: (e) => busy(e.currentTarget, async () => {
      await call("PUT", "/avatars/default", { name });
      toast(`${name} is now the default`);
      await refreshAvatars();
    }) }, "Make default"));
  }
  return el("li", { class: "job" },
    el("div", { class: "row" }, el("strong", {}, name), name === def ? el("span", { class: "pill done" }, "default") : null),
    el("div", { class: "sub" }, states.join(" · ")),
    el("div", { class: "actions" }, actions),
    status);
}

async function addAvatarPhotos(name, input, button, status, have) {
  const files = [...input.files];
  input.value = "";
  if (!files.length || button.disabled) return;
  if (have + files.length > MAX_AVATAR_IMAGES) {
    status.textContent = `an avatar holds at most ${MAX_AVATAR_IMAGES} photos (${name} has ${have})`;
    status.classList.add("err");
    return;
  }
  status.classList.remove("err");
  button.disabled = true;
  try {
    const keys = [];
    for (const [i, f] of files.entries()) {
      keys.push(await uploadImage(f, (t) => { status.textContent = `photo ${i + 1}/${files.length}: ${t}`; }));
    }
    status.textContent = "saving…";
    await call("POST", `/avatars/${encodeURIComponent(name)}/images`, { upload_keys: keys });
    status.textContent = "";
    toast(`Added ${files.length} photo${files.length > 1 ? "s" : ""} to ${name}`);
    await refreshAvatars();
  } catch (e) {
    if (!(e instanceof Unauthorized)) {
      status.textContent = e.message;
      status.classList.add("err");
    }
  } finally {
    button.disabled = false;
  }
}

async function refreshAvatars() {
  const ok = await loadAvatars();
  if (!ok) { // keep what is shown; say it may be out of date
    $("avatars-empty").hidden = true;
    return;
  }
  const list = avatarList.avatars.filter((a) => a && typeof a.name === "string");
  $("avatars-list").replaceChildren(...list.map((a) => avatarCard(a, avatarList.default)));
  $("avatars-empty").hidden = list.length > 0;
}

// ---------------------------------------------------------------- create

const MAX_IMAGE_BYTES = 25 * 1024 * 1024; // the API's limit for avatar and photo inputs
const AS_IS = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
const OUT_NAME = /^[A-Za-z0-9](?:[A-Za-z0-9._ -]{0,126}[A-Za-z0-9_-])?$/; // the API's meta.out_name rule
let avatarList = { default: null, avatars: [] };

/** A picked photo ready to upload: png/jpg/webp as they are; anything else Chrome can open (e.g. AVIF) becomes a
 *  JPEG; a photo it can't open (HEIC on most Android builds) is refused with what to do. */
export async function prepareImage(file) {
  if (file.size > 2 * MAX_IMAGE_BYTES) throw new Error(`the photo is larger than ${MAX_IMAGE_BYTES / 1024 / 1024} MB`);
  let blob = file;
  let ext = AS_IS[file.type];
  if (!ext) {
    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch (e) {
      const kind = (file.type || file.name.split(".").pop() || "this kind of file").replace("image/", "")
        .replace(/\+xml$/, "").toUpperCase();
      throw new Error(`this photo is ${kind}; set the camera to JPEG or pick another photo`);
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    bitmap.close();
    blob = await new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error("couldn't convert the photo"))),
                                                            "image/jpeg", 0.92));
    ext = "jpg";
  }
  if (blob.size > MAX_IMAGE_BYTES) throw new Error(`the photo is larger than ${MAX_IMAGE_BYTES / 1024 / 1024} MB`);
  return { blob, ext };
}

/** Upload one picked photo; returns its upload_key for a job or avatar. The same file is uploaded once: a retry
 *  after the job was refused reuses it. */
const uploaded = new WeakMap(); // File -> upload_key
async function uploadImage(file, progress) {
  if (uploaded.has(file)) return uploaded.get(file);
  progress("preparing the photo…");
  const { blob, ext } = await prepareImage(file);
  const up = await call("POST", "/uploads", { extension: ext });
  progress("uploading the photo…");
  await putUpload(up.put_url, blob);
  uploaded.set(file, up.upload_key);
  return up.upload_key;
}

async function loadAvatars() {
  let ok = false;
  try {
    const r = await call("GET", "/avatars");
    avatarList = { default: r.default || null, avatars: Array.isArray(r.avatars) ? r.avatars : [] };
    ok = true;
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(`Couldn't load the avatars: ${e.message}`);
  }
  for (const select of document.querySelectorAll("select[data-avatars]")) fillAvatars(select);
  return ok;
}

function fillAvatars(select) {
  const keep = select.value;
  const opts = [];
  const def = avatarList.default;
  opts.push(el("option", { value: "" }, def ? `Default (${def})` : "Default avatar"));
  if (select.dataset.avatars === "video") opts.push(el("option", { value: "none" }, "No avatar"));
  for (const a of avatarList.avatars) {
    if (a && typeof a.name === "string") opts.push(el("option", { value: a.name }, a.name));
  }
  if (select.dataset.avatars === "photos") opts.push(el("option", { value: "upload" }, "A photo from this phone"));
  select.replaceChildren(...opts);
  if ([...select.options].some((o) => o.value === keep)) select.value = keep;
  select.dispatchEvent(new Event("change"));
}

function showKind(kind) {
  for (const f of document.querySelectorAll("form[data-kind]")) f.hidden = f.dataset.kind !== kind;
  for (const b of document.querySelectorAll("#create-kinds button")) b.setAttribute("aria-pressed", b.dataset.kind === kind);
}

function outName(form) {
  const name = form.elements.out.value.trim();
  if (name && !OUT_NAME.test(name)) {
    throw new Error("file name: letters, digits, spaces and . _ - only (no folders), up to 128 characters");
  }
  return name || undefined;
}

function wholeNumber(value, what) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1) throw new Error(`${what}: a whole number`);
  return n;
}

function lines(text) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean);
}

async function jobBody(form, progress) {
  const body = await kindBody(form, progress);
  if (form.elements.realism_check.checked) body.realism_check = true;
  return body;
}

async function kindBody(form, progress) {
  const f = form.elements;
  const kind = form.dataset.kind;
  if (kind === "video") {
    const body = { type: "video", prompt: f.prompt.value.trim() };
    if (!body.prompt) throw new Error("describe the video");
    if (f.seconds.value) body.seconds = wholeNumber(f.seconds.value, "length");
    if (f.avatar.value) body.avatar = f.avatar.value;
    return body;
  }
  if (kind === "photos") {
    const scenes = lines(f.scenes.value);
    if (scenes.length > 8) throw new Error("at most 8 scenes");
    const body = { type: "photos", count: f.count.value ? wholeNumber(f.count.value, "how many") : scenes.length || 4 };
    if (scenes.length) body.scenes = scenes.map((prompt) => ({ prompt }));
    if (f.subject.value === "upload") {
      const file = f.photo.files[0];
      if (!file) throw new Error("pick a photo of the person");
      body.upload_key = await uploadImage(file, progress);
    } else if (f.subject.value) {
      body.avatar = f.subject.value;
    }
    return body;
  }
  const story = f.story.value.trim();
  if (!story) throw new Error("write or paste the story");
  return { type: "story", story, subtitles: f.subtitles.checked, narration: f.narration.checked };
}

async function submitCreate(form, button) {
  const status = form.querySelector(".form-status");
  const progress = (text) => { status.textContent = text; status.classList.remove("err"); };
  progress("");
  button.disabled = true;
  try {
    const out = outName(form);
    const body = await jobBody(form, progress);
    progress("sending…");
    const meta = { source: "phone" };
    if (out) meta.out_name = out;
    const r = await call("POST", "/jobs", { ...body, meta });
    progress("");
    toast(`Queued (${String(r.job_id || "").slice(0, 8)})`);
    const keepCheck = form.elements.realism_check.checked;
    form.reset();
    form.elements.realism_check.checked = keepCheck; // remembered, not reset
    for (const s of form.querySelectorAll("select[data-avatars]")) fillAvatars(s);
    show("history");
    refreshHistory().catch(staleNote);
  } catch (e) {
    if (!(e instanceof Unauthorized)) { // shown once, next to the button (the pairing screen covers a 401)
      status.textContent = e.message;
      status.classList.add("err");
    } else {
      progress("");
    }
  } finally {
    button.disabled = false;
  }
}

function setupCreate() {
  for (const b of document.querySelectorAll("#create-kinds button")) b.addEventListener("click", () => showKind(b.dataset.kind));
  for (const form of document.querySelectorAll("form[data-kind]")) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      submitCreate(form, form.querySelector("button[type=submit]"));
    });
  }
  const subject = $("photos-subject");
  subject.addEventListener("change", () => { $("photos-upload").hidden = subject.value !== "upload"; });
  showKind("video");
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
  stopPairing();
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
  setupCreate();
  setupRealismToggle();
  $("job-back").addEventListener("click", () => {
    leaveJob();
    show("history");
    refreshHistory().catch(staleNote);
  });
  for (const b of document.querySelectorAll("nav button")) {
    b.addEventListener("click", () => {
      leaveJob();
      show(b.dataset.view);
      if (b.dataset.view === "history") refreshHistory().catch(staleNote);
      if (b.dataset.view === "create") loadAvatars();
      if (b.dataset.view === "avatars") refreshAvatars();
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
