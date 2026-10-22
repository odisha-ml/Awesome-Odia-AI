---
name: check-awesome-odia-ai
description: Checks the Awesome Odia AI list in README.md. The gate covers the entry format, duplicate entries and the table of contents. A separate command checks for dead links. Use after editing README.md or scripts/check_list.py, and before opening or updating a pull request.
---

# Check Awesome Odia AI

Run everything from the repository root. The rules live in `AGENTS.md`.

## 1. Doctor

```bash
python3 --version && node --version && curl -sI https://github.com | head -1
```

The list check needs only Python 3. The link check also needs Node 22 or newer and network access.

## 2. Run

```bash
scripts/check.sh
scripts/check.sh links
```

1. `scripts/check.sh` is the gate. It runs `python3 scripts/check_list.py README.md`, which ends with `LIST OK: <n> entries` or `LIST FAIL: <n> problem(s)`. Each problem line starts with `README.md:<line>:`.
2. `scripts/check.sh links` runs `markdown-link-check@3.14.2` with `.markdown-link-check.json` over every link and anchor, then prints `LINKS OK`. In CI a failure here does not block a PR. `AGENTS.md` lists the skipped hosts and why.

## 3. Fix what failed

| Output | Fix |
| --- | --- |
| `entries start with '- ' at column 0` | Use `- `, not `*` or an indent. |
| `entry has no http(s) link` | Link the resource. |
| `empty link target '()'` | Add the URL or delete the empty link. |
| `duplicate entry '<name>', first on line <n>` | Keep one entry. Move details from the other into its description. |
| `table of contents does not match the headings` | Replace the list inside `<details>` with the printed block. |
| `heading has trailing spaces` | Trim the heading. |
| `table of contents not found` | Restore the `<details>` and `</details>` lines around the table of contents. |
| `entry starts with '[' but has no '](` | Close the name link as `[Name](url)`. |
| `[✖] <url> → Status: 404`, `401` or `403` | Replace the link with the resource's new home or an archive.org snapshot. If there is none, remove the entry. |
| `[✖] <url> → Status: 0` | No answer. This is often your network. Open the URL from another network before you change anything. |

## 4. Evidence

Paste the full output of both commands into the PR body. If some links fail only from your network, list them and say how you confirmed each one.

## 5. When you change check_list.py

Prove each rule still fails on a broken copy, and leave `README.md` untouched:

```bash
cp README.md /tmp/list.md
sed -i 's/^- \[Odia-ARC\]/* [Odia-ARC]/' /tmp/list.md
python3 scripts/check_list.py /tmp/list.md
```

Expect `LIST FAIL`. Repeat with a duplicated entry line, a deleted table-of-contents line, an entry with no link, a deleted `<details>` line and an entry whose name link has no `](`. Remove `/tmp/list.md` when done.
