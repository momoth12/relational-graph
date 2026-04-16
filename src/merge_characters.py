"""Merge and deduplicate extracted characters into canonical character sheets.

Takes per-chapter raw character data and produces a unified list of characters
with canonical names, merged aliases, and chapter presence tracking.
Uses GPT-4o to resolve ambiguous name matches.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MERGE_PROMPT = """You are a literary analyst. You have a list of character mentions extracted
from different chapters of Émile Zola's "La Fortune des Rougon".

Your task is to deduplicate these characters and produce a canonical list.

Rules:
1. Each real character should appear EXACTLY ONCE with their most complete canonical name
   (e.g., "Pierre Rougon", not just "Pierre" or "Rougon")
2. Merge all aliases: different names/references for the same person go into "aliases"
3. Merge descriptions from all chapters into one coherent description
4. Track which chapters each character appears in ("chapters" field)
5. Characters referred to only by role/title (e.g., "le commandant") should keep that as
   canonical name UNLESS you can identify them as a named character
6. Use your knowledge of the novel to resolve ambiguities (e.g., "le docteur" = "Pascal Rougon")
7. "family_branch" should be one of: "Rougon", "Macquart", "Mouret", "other"

Input format: a dict mapping chapter keys to arrays of character objects.

You MUST return a JSON object with a single key "characters" containing an array of canonical character objects.

Example:
{"characters": [
  {
    "canonical_name": "Pierre Rougon",
    "aliases": ["Rougon", "Pierre", "le père Rougon", "M. Rougon"],
    "description": "...",
    "family_branch": "Rougon",
    "chapters": ["preface", "1", "3", "5"]
  }
]}

IMPORTANT: Each entry MUST have "canonical_name" (string), "aliases" (list of strings), "description" (string), "family_branch" (string), "chapters" (list of strings). No markdown fences, no explanation.
"""


def merge_characters(
    client: OpenAI, raw_characters: dict, cache_dir: Path
) -> list[dict]:
    """Merge raw per-chapter characters into canonical list via LLM."""
    cache_file = cache_dir / "merged_characters.json"
    if cache_file.exists():
        print("Using cached merged characters")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    print("Calling GPT-4o to merge and deduplicate characters...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": MERGE_PROMPT},
            {"role": "user", "content": json.dumps(raw_characters, ensure_ascii=False)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)

    # Extract the characters list from the response object
    if isinstance(parsed, dict):
        parsed = parsed.get("characters", [])
        if not isinstance(parsed, list):
            parsed = []
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected LLM response format: {raw[:200]}")

    # Validate structure and deduplicate by canonical_name
    seen: dict[str, dict] = {}
    for char in parsed:
        if "canonical_name" not in char:
            raise ValueError(f"Missing canonical_name in character: {char}")
        char.setdefault("aliases", [])
        char.setdefault("description", "")
        char.setdefault("family_branch", "other")
        char.setdefault("chapters", [])

        name = char["canonical_name"]
        if name in seen:
            # Merge duplicate into existing entry
            existing = seen[name]
            existing["aliases"] = list(set(existing["aliases"] + char["aliases"]))
            existing["chapters"] = list(set(existing["chapters"] + char["chapters"]))
            if char["description"] and len(char["description"]) > len(existing["description"]):
                existing["description"] = char["description"]
        else:
            seen[name] = char
    parsed = list(seen.values())

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def main(input_path: str, output_path: str, cache_dir: str = "data/cache") -> list[dict]:
    raw_characters = json.loads(Path(input_path).read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    characters = merge_characters(client, raw_characters, Path(cache_dir))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Canonical characters: {len(characters)}")
    for c in characters:
        aliases_str = ", ".join(c["aliases"][:3])
        if len(c["aliases"]) > 3:
            aliases_str += "..."
        print(f"  {c['canonical_name']} [{c['family_branch']}] (aliases: {aliases_str})")

    return characters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw characters into canonical list")
    parser.add_argument("--input", default="data/raw_characters.json", help="Raw characters JSON")
    parser.add_argument("--output", default="data/characters.json", help="Output canonical JSON")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    args = parser.parse_args()
    main(args.input, args.output, args.cache_dir)
