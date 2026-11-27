// Request signing for the phone page: the same scheme as `cli/vidgen.py` sign() and `ssh-keygen -Y sign`
// (an OpenSSH SSHSIG v1, namespace vidgen-api, sha512) made with an ed25519 key that lives only in this browser.
// The private key is created non-extractable and kept in IndexedDB. Format: lambda/api/handler.py "auth".
// docs/specs/2026-11-27-phone-ui.md

const NAMESPACE = "vidgen-api";
const HASH = "sha512";
const enc = new TextEncoder();

function concat(...parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

function u32(n) {
  return new Uint8Array([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]);
}

function sshString(bytes) {
  const b = typeof bytes === "string" ? enc.encode(bytes) : bytes;
  return concat(u32(b.length), b);
}

function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function hex(bytes) {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Wire-format ssh-ed25519 public key blob for a raw 32-byte key. */
export function publicKeyBlob(pubRaw) {
  return concat(sshString("ssh-ed25519"), sshString(pubRaw));
}

export const COMMENT = /^[A-Za-z0-9._@-]{1,64}$/;

/** The authorized_keys line to paste into authorized_ssh_public_keys (the comment names the client). */
export function publicKeyLine(pubRaw, comment) {
  if (comment && !COMMENT.test(comment)) throw new Error("name: 1-64 letters, digits or . _ @ -");
  return `ssh-ed25519 ${b64(publicKeyBlob(pubRaw))}${comment ? " " + comment : ""}`;
}

/** The path exactly as fetch() sends it (dot segments resolved, characters percent-encoded), without the
 *  query or fragment: what the API sees as rawPath. */
export function wirePath(path) {
  return new URL(path, "https://api.invalid").pathname;
}

/** The canonical string the API verifies: method, path (no query), timestamp, nonce, sha256 of the body. */
export async function canonical(method, path, ts, nonce, bodyBytes) {
  const bodyHash = hex(new Uint8Array(await crypto.subtle.digest("SHA-256", bodyBytes)));
  return ["vidgen-v1", method.toUpperCase(), wirePath(path), String(ts), nonce, bodyHash].join("\n");
}

/** An armored SSHSIG over message, byte-identical to `ssh-keygen -Y sign -n vidgen-api` with the same key. */
export async function sshsig(privateKey, pubRaw, message) {
  const msg = typeof message === "string" ? enc.encode(message) : message;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-512", msg));
  const signed = concat(enc.encode("SSHSIG"), sshString(NAMESPACE), sshString(""), sshString(HASH),
                        sshString(digest));
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, privateKey, signed));
  const blob = concat(enc.encode("SSHSIG"), u32(1), sshString(publicKeyBlob(pubRaw)), sshString(NAMESPACE),
                      sshString(""), sshString(HASH), sshString(concat(sshString("ssh-ed25519"), sshString(sig))));
  const body = b64(blob).match(/.{1,70}/g).join("\n");
  return `-----BEGIN SSH SIGNATURE-----\n${body}\n-----END SSH SIGNATURE-----\n`;
}

/** The three x-vidgen-* headers for one request (ts and nonce are for tests; normally now and random). */
export async function signedHeaders(key, method, path, bodyBytes, ts, nonce) {
  ts = ts ?? Math.floor(Date.now() / 1000);
  nonce = nonce ?? hex(crypto.getRandomValues(new Uint8Array(16)));
  const armored = await sshsig(key.privateKey, key.pubRaw, await canonical(method, path, ts, nonce, bodyBytes));
  return { "x-vidgen-timestamp": String(ts), "x-vidgen-nonce": nonce, "x-vidgen-signature": btoa(armored) };
}

// ---------------------------------------------------------------- key storage (browser only)

const DB = "vidgen";
const STORE = "keys";
const KEY_ID = "phone";

/** Run fn(store) in one transaction; resolves with the last request's result once the transaction commits. */
function idb(mode, fn) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore(STORE);
    open.onerror = () => reject(open.error);
    open.onblocked = () => reject(new Error("close other vidgen tabs and reload"));
    open.onsuccess = () => {
      const db = open.result;
      db.onversionchange = () => db.close();
      let req;
      try {
        const tx = db.transaction(STORE, mode);
        tx.oncomplete = () => { db.close(); resolve(req && req.result); };
        tx.onerror = tx.onabort = () => { db.close(); reject(tx.error || new Error("storage failed")); };
        req = fn(tx.objectStore(STORE), tx);
      } catch (e) {
        db.close();
        reject(e);
      }
    };
  });
}

/** This phone's key ({privateKey, pubRaw}), or null before the first visit. */
export async function loadKey() {
  return (await idb("readonly", (s) => s.get(KEY_ID))) || null;
}

/** This phone's key, created on first use. The private key can't be exported, not even by this page. Two tabs
 *  opening at once end up with the same stored key: add() refuses to overwrite, and the loser reads the winner's. */
export async function loadOrCreateKey() {
  const existing = await loadKey();
  if (existing) return existing;
  const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, false, ["sign"]);
  const pubRaw = new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey));
  try {
    await idb("readwrite", (s) => s.add({ privateKey: pair.privateKey, pubRaw }, KEY_ID));
  } catch (e) {
    if (e && e.name !== "ConstraintError") throw e;
  }
  return loadKey();
}

export class Unauthorized extends Error {}
export class Offline extends Error {}

/** A signed call to this API; JSON in and out. Throws Unauthorized on 401 (not paired, or the clock is off),
 *  Offline when the request can't be sent, Error with the API's message otherwise. */
export async function signedFetch(key, method, path, body) {
  method = method.toUpperCase();
  const url = new URL(path, location.origin);
  if (url.origin !== location.origin) throw new Error("only this API can be called");
  const data = body === undefined ? new Uint8Array() : enc.encode(JSON.stringify(body));
  const headers = { "content-type": "application/json", ...(await signedHeaders(key, method, url.pathname, data)) };
  let r;
  try {
    r = await fetch(url.pathname + url.search, { method, headers, body: data.length ? data : undefined });
  } catch (e) {
    throw new Offline("can't reach the server; check the connection");
  }
  if (r.status === 401) {
    const server = Date.parse(r.headers.get("date") || "");
    const skew = Number.isFinite(server) ? Math.abs(server - Date.now()) / 1000 : 0;
    throw new Unauthorized(skew > 240 ? "this phone's clock is off; set the time automatically and retry"
                                      : "this phone isn't paired");
  }
  let out = null;
  try {
    out = await r.json();
  } catch (e) {
    if (r.ok) throw new Error(`unexpected answer from the server (HTTP ${r.status})`);
  }
  if (!r.ok) throw new Error(out && typeof out.error === "string" ? out.error : `HTTP ${r.status}`);
  return out && typeof out === "object" ? out : {};
}
