#!/usr/bin/env python3
"""
Extract only the AI model response from markdown files in `responses/` (non-recursive).

Assumption: The response begins immediately after the first line that starts with
"## Model:" and continues to the end of the file.

Outputs cleaned files to `responses_clean/` with filenames prefixed by a sortable
date/order derived from the source files in `responses/`.
"""

from datetime import datetime
from pathlib import Path
import re
import sys


def extract_response(text: str) -> str | None:
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Find the line that starts with '## Model:'
    m = re.search(r"^##\s*Model:.*$", text, flags=re.MULTILINE)
    if not m:
        return None
    # Slice everything after that line
    start = text.find("\n", m.end())
    if start == -1:
        # '## Model:' is the last line; nothing to extract
        return ""
    extracted = text[start + 1 :]
    # Trim leading/trailing whitespace but keep internal structure
    return extracted.strip() + "\n"


def extract_prompt_title(text: str) -> str | None:
    # Prefer the explicit prompt line if present
    m = re.search(r"^#\s*Prompt:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    # Fallback: first non-empty line
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def slugify(text: str, max_len: int = 80) -> str:
    # Keep ASCII only and turn non-word runs into single dashes
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub(r"[^\w]+", "-", text)
    text = text.strip("-")
    if not text:
        return "untitled"
    return text[:max_len].rstrip("-")


def sortable_prefix(md: Path) -> str:
    # Use mtime to preserve the original date-modified ordering.
    mtime_ns = md.stat().st_mtime_ns
    mtime_sec = mtime_ns / 1_000_000_000
    dt = datetime.fromtimestamp(mtime_sec)
    return f"{dt:%Y%m%d-%H%M%S}-{mtime_ns % 1_000_000_000:09d}"


def main() -> int:
    src_dir = Path("responses")
    if not src_dir.is_dir():
        print("Error: 'responses' directory not found.", file=sys.stderr)
        return 1

    out_dir = Path("responses_clean")
    out_dir.mkdir(exist_ok=True)

    md_files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md")

    processed = 0
    skipped = 0
    for md in md_files:
        try:
            text = md.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skip {md.name}: read error: {e}", file=sys.stderr)
            skipped += 1
            continue

        extracted = extract_response(text)
        if extracted is None:
            print(f"Skip {md.name}: '## Model:' marker not found.")
            skipped += 1
            continue

        order_prefix = sortable_prefix(md)
        base_name = f"{order_prefix}__{md.name}"
        out_path = out_dir / base_name
        if out_path.exists():
            i = 1
            while (out_dir / f"{order_prefix}__{i}__{md.name}").exists():
                i += 1
            out_path = out_dir / f"{order_prefix}__{i}__{md.name}"
        try:
            out_path.write_text(extracted, encoding="utf-8")
            processed += 1
        except Exception as e:
            print(f"Skip {md.name}: write error: {e}", file=sys.stderr)
            skipped += 1

    print(f"Done. Processed: {processed}, Skipped: {skipped}, Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
