#!/bin/sh
set -e
cd "$(dirname "$0")/.."
python3 scripts/check_list.py README.md
npx --yes markdown-link-check@3.14.2 --quiet README.md
echo "LINKS OK"
