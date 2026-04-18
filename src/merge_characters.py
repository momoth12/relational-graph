# Merging  per-chapter characters into a canonical deduplicated list


import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MERGE_PROMPT_TEMPLATE = """You are a literary analyst. You have a list of character mentions extracted
from different chapters of "{title}" by {author}.

Your task is to deduplicate these characters and produce a canonical list.

Rules:
1. Each real character should appear EXACTLY ONCE with their most complete canonical name
2. Merge all aliases: different names/references for the same person go into "aliases"
3. Merge descriptions from all chapters into one coherent description
4. Track which chapters each character appears in ("chapters" field)
5. Characters referred to only by role/title should keep that as canonical name
   UNLESS you can identify them as a named character
6. Use your knowledge of the novel to resolve ambiguities
7. "group" should be one of: {groups}

Input format: a dict mapping chapter keys to arrays of character objects.

You MUST return a JSON object with a single key "characters" containing an array of canonical character objects.

Example:
{{"characters": [
  {{
    "canonical_name": "Jean Dupont",
    "aliases": ["Dupont", "Jean", "M. Dupont"],
    "description": "...",
    "group": "...",
    "chapters": ["1", "3", "5"]
  }}
]}}

IMPORTANT: Each entry MUST have "canonical_name" (string), "aliases" (list of strings), "description" (string), "group" (string), "chapters" (list of strings). No markdown fences, no explanation.
"""


def _load_book_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_characters(
    client: OpenAI, raw_characters: dict, cache_dir: Path,
    system_prompt: str = "",
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(raw_characters, ensure_ascii=False)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)

    if isinstance(parsed, dict):
        parsed = parsed.get("characters", [])
        if not isinstance(parsed, list):
            parsed = []
    if not isinstance(parsed, list):
        raise ValueError(f"Unexpected LLM response format: {raw[:200]}")

    seen: dict[str, dict] = {}
    for char in parsed:
        if "canonical_name" not in char:
            raise ValueError(f"Missing canonical_name in character: {char}")
        char.setdefault("aliases", [])
        char.setdefault("description", "")
        char.setdefault("group", "other")
        char.setdefault("chapters", [])
        if "family_branch" in char and "group" not in char:
            char["group"] = char.pop("family_branch")
        elif "family_branch" in char:
            char.pop("family_branch")

        name = char["canonical_name"]
        if name in seen:
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


def main(input_path: str, output_path: str, cache_dir: str = "data/cache",
         book_config_path: str = "book_config.json") -> list[dict]:
    raw_characters = json.loads(Path(input_path).read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    book_config = _load_book_config(book_config_path)
    groups = ', '.join(f'"{g}"' for g in book_config.get("groups", [])) + ', "other"'
    system_prompt = MERGE_PROMPT_TEMPLATE.format(
        title=book_config["title"], author=book_config["author"], groups=groups
    )

    characters = merge_characters(client, raw_characters, Path(cache_dir), system_prompt)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Canonical characters: {len(characters)}")
    for c in characters:
        aliases_str = ", ".join(c["aliases"][:3])
        if len(c["aliases"]) > 3:
            aliases_str += "..."
        print(f"  {c['canonical_name']} [{c.get('group', 'other')}] (aliases: {aliases_str})")

    return characters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw characters into canonical list")
    parser.add_argument("--input", default="data/raw_characters.json", help="Raw characters JSON")
    parser.add_argument("--output", default="data/characters.json", help="Output canonical JSON")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    parser.add_argument("--book-config", default="book_config.json", help="Book configuration JSON")
    args = parser.parse_args()
    main(args.input, args.output, args.cache_dir, args.book_config)
