// Node helper for tests/test_sign_parity.py: signs one request with sign.js and prints the headers as JSON.
// args: <pkcs8 base64> <pub raw base64> <method> <path> <ts> <nonce> <body>
import { publicKeyLine, signedHeaders } from "../lambda/api/phone/sign.js";

const [pkcs8, pub, method, path, ts, nonce, body] = process.argv.slice(2);
const der = Buffer.from(pkcs8, "base64");
const privateKey = await crypto.subtle.importKey("pkcs8", der, { name: "Ed25519" }, false, ["sign"]);
const key = { privateKey, pubRaw: new Uint8Array(Buffer.from(pub, "base64")) };
const headers = await signedHeaders(key, method, path, new TextEncoder().encode(body), Number(ts), nonce);
console.log(JSON.stringify({ headers, line: publicKeyLine(key.pubRaw, "phone") }));
