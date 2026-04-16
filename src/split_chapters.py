"""Split La Fortune des Rougon into chapters.

Reads the source text and splits it into Preface + 7 chapters,
outputting a JSON file with chapter keys and text values.
"""

import argparse
import json
import re
from pathlib import Path


CHAPTER_PATTERN = re.compile(r"^(I{1,3}|IV|VI{0,2}|VII)$")


def split_chapters(text: str) -> dict[str, str]:
    """Split the novel text into named chapters.

    Returns a dict like {"preface": "...", "1": "...", ..., "7": "..."}.
    """
    lines = text.splitlines(keepends=True)

    # Find the PRÉFACE heading (actual chapter start, not the TOC one)
    preface_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "PRÉFACE" and i > 80:
            # Skip the TOC entry; the actual preface heading is after line 80
            preface_start = i
            break

    if preface_start is None:
        raise ValueError("Could not find PRÉFACE heading in the text")

    # Find chapter headings (roman numerals on their own line, after the TOC)
    chapter_starts: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > preface_start and CHAPTER_PATTERN.match(stripped):
            chapter_num = _roman_to_int(stripped)
            chapter_starts.append((str(chapter_num), i))

    if len(chapter_starts) != 7:
        raise ValueError(
            f"Expected 7 chapter headings, found {len(chapter_starts)}: "
            f"{[c[1] for c in chapter_starts]}"
        )

    # Find the end of the novel text (FIN marker)
    fin_line = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "FIN":
            fin_line = i
            break

    # Build sections
    boundaries = [("preface", preface_start)] + chapter_starts
    chapters = {}
    for idx, (key, start) in enumerate(boundaries):
        if idx + 1 < len(boundaries):
            end = boundaries[idx + 1][1]
        else:
            end = fin_line
        # Skip the heading line itself and any blank lines right after
        content_start = start + 1
        chapter_text = "".join(lines[content_start:end]).strip()
        chapters[key] = chapter_text

    return chapters


def _roman_to_int(s: str) -> int:
    roman_map = {"I": 1, "V": 5, "X": 10}
    result = 0
    prev = 0
    for ch in reversed(s):
        val = roman_map[ch]
        if val < prev:
            result -= val
        else:
            result += val
        prev = val
    return result


def main(input_path: str, output_path: str) -> dict[str, str]:
    text = Path(input_path).read_text(encoding="utf-8")
    chapters = split_chapters(text)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Split into {len(chapters)} sections:")
    for key, text in chapters.items():
        print(f"  {key}: {len(text)} chars, {len(text.splitlines())} lines")

    return chapters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split novel text into chapters")
    parser.add_argument("--input", required=True, help="Path to the source .txt file")
    parser.add_argument("--output", default="data/chapters.json", help="Output JSON path")
    args = parser.parse_args()
    main(args.input, args.output)
