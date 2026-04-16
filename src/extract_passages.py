"""Extract text passages that describe relationships between characters.

Sends each chapter + canonical character list to GPT-4o and asks it to
identify and extract only the passages where a relationship is being
described, implied, or revealed. This provides focused input for the
relationship extraction step.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

SYSTEM_PROMPT = """You are a literary analyst specializing in 19th-century French literature.
You will receive a chapter from Émile Zola's "La Fortune des Rougon" along with a list of
canonical character names.

Your task is to extract ONLY the passages from the text where a relationship between
two or more characters is being described, implied, or revealed.

A "relationship passage" includes:
- Family ties being stated ("son fils Pierre", "sa mère Adélaïde")
- Romantic or marital bonds ("sa femme Félicité", "Silvère aimait Miette")
- Political alliances or oppositions
- Friendships, enmities, professional ties
- Any interaction that reveals the nature of a bond between characters

For each passage, provide:
- "text": the exact quote from the chapter (keep it short but complete enough to understand the relationship, max ~200 chars)
- "characters": list of canonical names involved (use EXACT names from the provided list)
- "relationship_hint": a brief label for what type of relationship is described
  (e.g., "parent-child", "spouses", "political allies", "lovers", "siblings", "enemies")

You MUST return a JSON object with a single key "passages" containing an array of passage objects.

Example:
{"passages": [
  {"text": "Pierre, le fils légitime d'Adélaïde, grandit dans la haine", "characters": ["Adélaïde Fouque", "Pierre Rougon"], "relationship_hint": "parent-child"},
  {"text": "Félicité, sa femme, une petite femme sèche", "characters": ["Pierre Rougon", "Félicité Rougon"], "relationship_hint": "spouses"}
]}

Each entry MUST have "text" (string), "characters" (list of strings), and "relationship_hint" (string).

IMPORTANT:
- Extract the ACTUAL text from the chapter, do not paraphrase
- Use ONLY canonical names from the provided list for the "characters" field
- If a passage involves a character not in the list, still extract it but use the name as it appears in the text
- Be thorough: extract ALL passages that reveal or imply a relationship
- Do NOT extract passages that merely mention a character without relationship context
"""


def _call_llm(client: OpenAI, chapter_key: str, chapter_text: str, char_list_str: str, retries: int = 2) -> list[dict]:
    """Single LLM call for passage extraction with retry. Returns validated list."""
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Canonical character list:\n{char_list_str}\n\n"
                        f"Chapter: {chapter_key}\n\n{chapter_text}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=16384,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            print(f"  [{chapter_key}] WARNING: empty response (attempt {attempt + 1}/{retries + 1})")
            if attempt < retries:
                continue
            return []

        raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [{chapter_key}] WARNING: invalid JSON (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                continue
            return []

        if isinstance(parsed, dict):
            parsed = parsed.get("passages", [])
            if not isinstance(parsed, list):
                parsed = []
        if not isinstance(parsed, list):
            return []

        validated = []
        for item in parsed:
            if isinstance(item, dict) and "text" in item:
                item.setdefault("characters", [])
                item.setdefault("relationship_hint", "unknown")
                validated.append(item)
        return validated

    return []


def extract_passages_from_chapter(
    client: OpenAI,
    chapter_key: str,
    chapter_text: str,
    characters: list[dict],
    cache_dir: Path,
) -> list[dict]:
    """Extract relationship passages from a single chapter, with caching."""
    cache_file = cache_dir / f"passages_{chapter_key}.json"
    if cache_file.exists():
        print(f"  [{chapter_key}] Using cached result")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    char_names = [c["canonical_name"] for c in characters]
    char_list_str = "\n".join(f"- {name}" for name in char_names)

    # Split long chapters into chunks of ~30k chars to avoid output token truncation
    CHUNK_SIZE = 30000
    if len(chapter_text) > CHUNK_SIZE:
        # Determine number of parts needed
        num_parts = (len(chapter_text) // CHUNK_SIZE) + 1
        # Split at paragraph breaks
        chunks = []
        remaining = chapter_text
        for i in range(num_parts - 1):
            target = len(remaining) // (num_parts - i)
            split_pos = remaining.rfind("\n\n", 0, target + 500)
            if split_pos == -1:
                split_pos = remaining.rfind("\n", 0, target + 500)
            if split_pos == -1:
                split_pos = target
            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        chunks.append(remaining)

        print(f"  [{chapter_key}] Splitting ({len(chapter_text)} chars) into {len(chunks)} parts")
        parsed = []
        for i, chunk in enumerate(chunks):
            print(f"  [{chapter_key}] Calling GPT-4o (part {i+1}/{len(chunks)})...")
            parsed.extend(_call_llm(client, f"{chapter_key} (part {i+1})", chunk, char_list_str))
    else:
        print(f"  [{chapter_key}] Calling GPT-4o...")
        parsed = _call_llm(client, chapter_key, chapter_text, char_list_str)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def main(
    chapters_path: str,
    characters_path: str,
    output_path: str,
    cache_dir: str = "data/cache",
) -> dict[str, list[dict]]:
    chapters = json.loads(Path(chapters_path).read_text(encoding="utf-8"))
    characters = json.loads(Path(characters_path).read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = Path(cache_dir)

    all_passages: dict[str, list[dict]] = {}
    total = 0
    for key, text in tqdm(chapters.items(), desc="Extracting passages", unit="ch"):
        passages = extract_passages_from_chapter(client, key, text, characters, cache)
        all_passages[key] = passages
        total += len(passages)
        tqdm.write(f"  [{key}] Found {len(passages)} passages")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_passages, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total: {total} relationship passages across {len(chapters)} chapters")
    return all_passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relationship passages using GPT-4o")
    parser.add_argument("--chapters", default="data/chapters.json", help="Chapters JSON")
    parser.add_argument("--characters", default="data/characters.json", help="Canonical characters JSON")
    parser.add_argument("--output", default="data/passages.json", help="Output passages JSON")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    args = parser.parse_args()
    main(args.chapters, args.characters, args.output, args.cache_dir)
