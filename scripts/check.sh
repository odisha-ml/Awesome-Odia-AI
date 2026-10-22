#!/bin/sh
set -e
cd "$(dirname "$0")/.."
if [ "$1" = links ]; then
  npx --yes markdown-link-check@3.14.2 --quiet --config .markdown-link-check.json README.md
  echo "LINKS OK"
else
  python3 scripts/check_list.py README.md
fi
