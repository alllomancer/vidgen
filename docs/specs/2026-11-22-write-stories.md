# Write stories from a premise (`vidgen write`)

Jira: N/A
Date: 2026-11-22
Status: Approved
Last reviewed: 2026-11-22

## Problem

`story` films a story you already have. The model on the render box that plans those stories
(Qwen2.5-VL-7B-Instruct) can also write one from a premise, but there is no way to get the text
on its own to read or edit before filming.

## Approach

A text-only `write` job using the planner's model (no new model, no video): premise in, story.txt
out. CLI `vidgen write`, and a Write form in the dashboard's Create tab, with **Use in Story** in
History to load the result into the Story form.

## Design

- API `type: write`: `premise` (1-2000 chars), `words` (100-2000, default 800), `genre` (optional,
  <= 200 chars), `seed` (optional). Summary: the premise, words, genre.
- Worker `run_write` -> `models.write_story`: one chat call asking for "Title: …", a blank line,
  then the story; written for narration and filming (visual scenes, few named characters, clear
  arc). Sampled (temperature 0.8, top-p 0.9, repetition penalty 1.05) with `torch.manual_seed(seed)`;
  `max_new_tokens = min(4096, words * 1.8 + 100)`. `split_title` takes the title line (or a short
  first line). Uploads `story.txt` (title, blank line, story); result `{title, words, seed}`.
- CLI: `vidgen write PREMISE|- [--words N] [--genre G] [--seed S] [-o NAME.txt]`; downloads to
  `<name>.txt` (default `<job_id>.txt`); the watcher notifies "Story written" and logs the
  `vidgen story <file>` command.
- Dashboard: Create > Write (premise, words, genre, seed, file name). History rows of downloaded
  write jobs get **Use in Story**: `GET /api/jobs/<id>/story` returns the local file's text (only
  for a downloaded write job, <= 20000 chars) and the page fills the Story form.

## Acceptance Criteria

- When `vidgen write "…" --words 600 -o x` finishes, `x.txt` shall hold a title line, a blank line
  and a story of roughly 600 words.
- If premise is empty or over 2000 chars, words outside 100-2000 or genre over 200 chars, then the
  API shall return 400.
- When the same premise, words, genre and seed are written twice, the text shall be the same.
- When Use in Story is clicked on a downloaded write job, the Story form shall contain the file's text.
- If the job is not a downloaded write job, then `/api/jobs/<id>/story` shall return 404.

## Out of Scope

A larger writing model; editing stories in the dashboard beyond the Story form; writing with
avatar descriptions; a content filter (deferred, as for the other jobs).

## Risks & Open Questions

- Quality: a 7B model writes simple, sometimes generic prose and may run short of the word target.
  (Measured in the live test; a bigger model is a separate decision.)
- Seed reproducibility across worker restarts with CPU offload. (Check live.)

## Testing

Unit tests (API validation, title split, worker run, CLI paths and download, dashboard text
endpoint). Live: one `vidgen write`, then Use in Story in the dashboard.

## Update: live check (2026-11-23)

`vidgen write "A lighthouse keeper finds a message in a bottle addressed to her, dated tomorrow."
--words 600 --genre "eerie, bittersweet" --seed 4`: downloaded as a titled story ("Message in a
Bottle"), 851 words (the model overshoots the target here; the runbook says lengths are
approximate), coherent with a clear arc, plain prose as expected from a 7B model. Use in Story
loaded the downloaded text into the Story form with `test-write-lighthouse.mp4` as the file name.
