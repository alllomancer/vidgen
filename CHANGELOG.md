# Changelog

## 2026-11-28

- Retry until it looks real: one picture or one 5 s clip, re-rendered from the best try's suggested prompt up to 25 times until the AI-look score is 0 or stops improving (dashboard, phone, `--realism-retries`).

## 2026-11-27

- Phone page for Android Chrome at `<api_url>/m`: History, Create (video, photos, story), job results with Save, Avatars, render box status; installable to the home screen. Its own ed25519 key signs requests like the CLI.
- Pair a phone from the dashboard's Phones panel (code + key fingerprint, approve on the Mac, remove any time); adding its key in Terraform still works.
- New avatars: choose which LoRAs to train (both, Krea 2, Wan, or none to upload your own): **LoRAs for this person** in the dashboard, `--loras` in the CLI.
- Upload an avatar's own Krea 2 and Wan LoRAs made elsewhere (avatar card **Upload LoRAs**, `vidgen avatar lora`); they're used right away.
- Phones panel shows a QR code of the phone page link.
- Optional AI-look check after a render (photos, videos, stories): a 0-10 score, the giveaways, and a suggested prompt and settings with Use suggestion (dashboard, phone) and `--realism-check` (CLI).
- Jobs carry `meta` (file name, avatar, source device); the watcher downloads jobs started on other devices; `vidgen jobs forget ID`.
