# Awesome Odia AI

`README.md` is a curated list of AI resources for the Odia language: models, datasets, tools, papers, events and communities. Almost every change adds, fixes or removes entries in it.

## Files

| Path | What it is |
| --- | --- |
| `README.md` | The list. |
| `scripts/check.sh` | `scripts/check.sh` runs the list check, the gate for every PR. `scripts/check.sh links` runs the link check. |
| `scripts/check_list.py` | The list check. It runs offline and checks the entry format, duplicates and the table of contents. |
| `.markdown-link-check.json` | Settings for the link check. |
| `.agents/skills/check-awesome-odia-ai/` | How to run the checks, read their output and prove they work. |
| `.github/workflows/deadlink_checker.yml` | Runs both checks on pushes to `main`, on pull requests, daily and on manual dispatch. |

## Entry format

Add each entry as one line:

```markdown
- [IndicParam](https://huggingface.co/datasets/bharatgenai/IndicParam) : Human-curated multiple-choice benchmark of 13,207 questions in 11 low-resource Indic languages, including Odia (CC-BY-NC-4.0). [[paper](https://arxiv.org/abs/2512.00333)]
```

The name links to the resource itself. The description says what it is, how much Odia it has, and the licence when it is known. Extra links go after the description as `[[paper](...)]`, `[[code](...)]` or `[[model](...)]`.

| Rule | Enforced by |
| --- | --- |
| Entries start with `- ` at column 0. | `check_list.py` |
| Every entry has an `http(s)` link, and no link target is empty. | `check_list.py` |
| An entry name appears once in the whole list, ignoring case. A resource that fits two sections goes in the main one, and the description names the other part (see OdiEnCorp 1.0). | `check_list.py` |
| The table of contents lists every `##` to `####` heading in order. On a mismatch the check prints the expected block to paste in. | `check_list.py` |
| Headings have no trailing spaces, so their anchors resolve. | `check_list.py` |
| Every link answers 200, 206 or 429. | `scripts/check.sh links` |

## Inclusion rules

These need judgment, so no script checks them.

- The resource covers Odia. Multilingual resources qualify only when Odia is one of the languages, and the description says so ("covering 14 Indic languages, including Odia").
- The linked page opens without signing in. A gated Hugging Face repo is fine when its model card is public. Write "gated" in the description. A link that answers 401 or 403, or redirects to a login page, is not fine. For example, `huggingface.co/OliveVoice/omni-odia-tts` answered 401 and was removed.
- Say "commercial" or "closed-source" when the data is not openly downloadable.
- Write plain descriptions. Leave out praise ("highly optimized", "paradigm-shifting") and download counts, which go stale.

## Section order

Top-level sections stay in this order: NLP, Audio, Computer Vision, Applications, Events, Community, then Contributing. Entries within a section are in the order they were added, not alphabetical. Append new entries at the end of their section and leave existing entries where they are. Add a new heading only when no existing section fits, and update the table of contents in the same change.

## Verify

```bash
scripts/check.sh
scripts/check.sh links
```

`scripts/check.sh` needs only `python3` and prints `LIST OK: <n> entries`. CI's `list` job runs it and fails the run when it fails.

`scripts/check.sh links` needs Node 22 or newer for `npx` and network access, and prints `LINKS OK`. CI's `links` job runs it, but a failure there does not fail a push or pull request, because sites go down for reasons unrelated to the change. The daily and manual runs do fail on a dead link. The README badge shows push runs only.

The link check retries a 429 three times and waits 20 seconds per link. It also accepts a 429 that persists, since that means rate limiting, not a dead page. It does not accept 403, which usually means a login wall. It skips these hosts:

| Host | Reason |
| --- | --- |
| `xlit.ai4bharat.org` | On 22 Oct 2026 it reset TLS for non-browser clients and answered 500 to a second fetcher. |
| `datasets.ai4bharat.org` | Same as `xlit.ai4bharat.org`. |
| `odishaaisummit.org` | Same as `xlit.ai4bharat.org`. |

Open a skipped host in a browser before you rely on it, and remove its pattern from `.markdown-link-check.json` once it answers a plain fetch. The skill in `.agents/skills/check-awesome-odia-ai/` explains each failure.

## Pull requests

- Keep one topic per PR, and name the added or removed resources in the title.
- Paste the output of `scripts/check.sh` and `scripts/check.sh links` into the PR body.
- Do not reformat or reorder entries you are not changing.
- Do not force-push a branch under review.
