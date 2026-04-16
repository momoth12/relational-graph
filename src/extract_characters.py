"""Extract characters from each chapter using GPT-4o.

Sends each chapter to the LLM and asks for structured character data.
Caches LLM responses per chapter to avoid redundant API calls.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = """You are a literary analyst specializing in 19th-century French literature.
You will receive a chapter from Émile Zola's "La Fortune des Rougon".

Extract ALL named characters mentioned in this chapter. For each character, provide:
- "name": the most complete form of their name as it appears in the text
- "aliases": list of other names, titles, or references used for this character in this chapter
  (e.g., "le docteur", "le père Rougon", first name only, etc.)
- "description": a brief description based on what this chapter reveals about them

You MUST return a JSON object with a single key "characters" containing an array of character objects.

Example:
{"characters": [
  {"name": "Pierre Rougon", "aliases": ["Rougon", "Pierre"], "description": "Ambitious merchant, head of the Rougon family"},
  {"name": "Silvère Mouret", "aliases": ["Silvère"], "description": "Young worker and republican insurgent"}
]}

If no named characters appear in the chapter, return: {"characters": []}

Important:
- Include ALL named characters, even if only briefly mentioned
- Use the most complete name form available in the text
- Do NOT invent information not present in the chapter
- Characters referred to only by title/role without a name should use that title as name
  (e.g., "le commandant", "le garde champêtre")
- Each entry MUST have "name" (string), "aliases" (list of strings), and "description" (string)
"""


def extract_characters_from_chapter(
    client: OpenAI, chapter_key: str, chapter_text: str, cache_dir: Path
) -> list[dict]:
    """Extract characters from a single chapter, with caching."""
    cache_file = cache_dir / f"characters_{chapter_key}.json"
    if cache_file.exists():
        print(f"  [{chapter_key}] Using cached result")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    print(f"  [{chapter_key}] Calling GPT-4o...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Chapter: {chapter_key}\n\n{chapter_text}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)

    # Extract the characters list from the response object
    if isinstance(parsed, dict):
        characters = parsed.get("characters", [])
        if not isinstance(characters, list):
            characters = []
    elif isinstance(parsed, list):
        characters = parsed
    else:
        raise ValueError(f"Unexpected LLM response format for {chapter_key}: {raw[:200]}")

    # Validate each entry is a proper character dict, not a flat string
    validated = []
    for item in characters:
        if isinstance(item, dict) and "name" in item:
            item.setdefault("aliases", [])
            item.setdefault("description", "")
            validated.append(item)
        elif isinstance(item, str):
            # LLM returned a flat string; skip it (bad response)
            print(f"  [{chapter_key}] WARNING: skipping flat string '{item}', expected object")
    parsed = validated

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def main(input_path: str, output_path: str, cache_dir: str = "data/cache") -> dict:
    chapters = json.loads(Path(input_path).read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = Path(cache_dir)

    raw_characters = {}
    for key, text in tqdm(chapters.items(), desc="Extracting characters", unit="ch"):
        chars = extract_characters_from_chapter(client, key, text, cache)
        raw_characters[key] = chars
        tqdm.write(f"  [{key}] Found {len(chars)} characters")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw_characters, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(v) for v in raw_characters.values())
    print(f"Total: {total} character mentions across {len(chapters)} chapters")
    return raw_characters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract characters from chapters using GPT-4o")
    parser.add_argument("--input", default="data/chapters.json", help="Chapters JSON path")
    parser.add_argument("--output", default="data/raw_characters.json", help="Output JSON path")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory for LLM responses")
    args = parser.parse_args()
    main(args.input, args.output, args.cache_dir)
