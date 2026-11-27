# Changelog

## 2026-11-27

- Phone page for Android Chrome at `<api_url>/m`: History, Create (video, photos, story), job results with Save, Avatars, render box status; installable to the home screen. Its own ed25519 key signs requests like the CLI.
- Pair a phone from the dashboard's Phones panel (code + key fingerprint, approve on the Mac, remove any time); adding its key in Terraform still works.
- Optional AI-look check after a render (photos, videos, stories): a 0-10 score, the giveaways, and a suggested prompt and settings with Use suggestion (dashboard, phone) and `--realism-check` (CLI).
- Jobs carry `meta` (file name, avatar, source device); the watcher downloads jobs started on other devices; `vidgen jobs forget ID`.
