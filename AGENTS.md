# Awesome Odia AI

`README.md` is a curated list of AI resources for the Odia language: models, datasets, tools, papers, events and communities. Almost every change adds, fixes or removes entries in it.

## Files

| Path | What it is |
| --- | --- |
| `README.md` | The list. |
| `scripts/check.sh` | The one check command. It runs the two checks below. |
| `scripts/check_list.py` | Offline check of the entry format, duplicates and the table of contents. |
| `.agents/skills/check-awesome-odia-ai/` | How to run the checks, read their output and prove they work. |
| `.github/workflows/deadlink_checker.yml` | Runs `scripts/check.sh` on pushes to `main`, on pull requests and daily. |

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
| Every link answers with a 2xx status. | `markdown-link-check` |

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
```

It needs `python3` and Node 20 or newer for `npx`. It prints `LIST OK: <n> entries`, then `LINKS OK`. Without network, run `python3 scripts/check_list.py`. The skill in `.agents/skills/check-awesome-odia-ai/` explains each failure.

## Pull requests

- Keep one topic per PR, and name the added or removed resources in the title.
- Paste the output of `scripts/check.sh` into the PR body.
- Do not reformat or reorder entries you are not changing.
- Do not force-push a branch under review.
