// vidgen phone page. Talks only to this API, signing every call with a key that never leaves this browser.
// docs/specs/2026-11-27-phone-ui.md

async function ed25519Supported() {
  try {
    await crypto.subtle.generateKey({ name: "Ed25519" }, false, ["sign"]);
    return true;
  } catch (e) {
    return false;
  }
}

async function main() {
  const status = document.getElementById("status");
  status.textContent = (await ed25519Supported())
    ? "Ready."
    : "This browser can't make the signing key this page needs. Update Chrome (version 137 or later).";
}

main();
