#!/usr/bin/env python3
import re
import sys
import unicodedata

HEADING = re.compile(r"^(#{2,4}) (.+)$")
ENTRY = re.compile(r"^- (\S.*)$")
LINK = re.compile(r"\[[^\]]*\]\(([^)]*)\)")


def slug(title, seen):
    kept = (c for c in title.strip().lower() if c in "-_ " or unicodedata.category(c)[0] in "LMN")
    base = "".join(kept).replace(" ", "-")
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def entry_name(text):
    if text.startswith("["):
        return text[1 : text.index("](")] if "](" in text else None
    return re.split(r":| \[\[", text, maxsplit=1)[0]


def check(lines):
    if "<details>" not in lines or "</details>" not in lines:
        return ["table of contents not found: expected a <details> block on its own lines"], 0
    errors = []
    toc_start = lines.index("<details>")
    toc_end = lines.index("</details>")
    toc = [line for line in lines[toc_start:toc_end] if line.lstrip().startswith("- [")]
    expected_toc = []
    seen_slugs = {}
    names = {}
    for number, line in enumerate(lines[toc_end:], toc_end + 1):
        heading = HEADING.match(line)
        if heading:
            depth = len(heading.group(1)) - 2
            title = heading.group(2)
            if title != title.strip():
                errors.append(f"{number}: heading has trailing spaces")
            expected_toc.append(f"{'  ' * depth}- [{title.strip()}](#{slug(title, seen_slugs)})")
            continue
        if not re.match(r"^\s*[-*+] ", line):
            continue
        entry = ENTRY.match(line)
        if not entry:
            errors.append(f"{number}: entries start with '- ' at column 0: {line.strip()[:60]}")
            continue
        targets = LINK.findall(line)
        if not any(target.startswith(("http://", "https://")) for target in targets):
            errors.append(f"{number}: entry has no http(s) link")
        if "" in targets:
            errors.append(f"{number}: empty link target '()'")
        name = entry_name(entry.group(1))
        if name is None:
            errors.append(f"{number}: entry starts with '[' but has no '](': {line.strip()[:60]}")
            continue
        name = name.strip().lower()
        if name in names:
            errors.append(f"{number}: duplicate entry '{name}', first on line {names[name]}")
        names.setdefault(name, number)
    if toc != expected_toc:
        errors.append("table of contents does not match the headings. Expected:\n" + "\n".join(expected_toc))
    return errors, len(names)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    with open(path, encoding="utf-8") as handle:
        errors, entries = check(handle.read().split("\n"))
    for error in errors:
        print(f"{path}:{error}")
    print(f"LIST FAIL: {len(errors)} problem(s)" if errors else f"LIST OK: {entries} entries")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
