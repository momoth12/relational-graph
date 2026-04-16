"""Extract structured relationships from pre-extracted passages using GPT-4o.

Receives relationship passages (from extract_passages.py) + canonical character list.
Normalizes edge direction (parent→child for asymmetric, alphabetical for symmetric).
Deduplicates across chapters.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

RELATIONSHIP_TYPES = [
    "parent_child",
    "spouse",
    "sibling",
    "lover",
    "grandparent_grandchild",
    "uncle_nephew",
    "friend",
    "enemy",
    "political_ally",
    "political_opponent",
    "employer_employee",
    "neighbor",
    "acquaintance",
]

DIRECTED_TYPES = {
    "parent_child",
    "grandparent_grandchild",
    "uncle_nephew",
    "employer_employee",
}

SYSTEM_PROMPT = """You are a literary analyst specializing in 19th-century French literature.
You will receive a chapter from Émile Zola's "La Fortune des Rougon" along with a list of
canonical character names.

Extract ALL relationships between characters that are mentioned or implied in this chapter.

For each relationship, provide:
- "source": canonical name of the first character (use EXACT names from the provided list)
- "target": canonical name of the second character (use EXACT names from the provided list)
- "type": one of: {types}
- "directed": true if the relationship is asymmetric (parent→child, employer→employee, etc.)
- "passage": a SHORT quote from the text that evidences this relationship (max 100 chars)
- "confidence": 0.0 to 1.0 — how confident you are this relationship exists

Direction rules for DIRECTED relationships:
- parent_child: source=parent, target=child
- grandparent_grandchild: source=grandparent, target=grandchild
- uncle_nephew: source=uncle/aunt, target=nephew/niece
- employer_employee: source=employer, target=employee

For SYMMETRIC relationships (spouse, sibling, lover, friend, enemy, etc.):
- Put the alphabetically first canonical name as "source"

You MUST return a JSON object with a single key "relationships" containing an array of relationship objects.

Example:
{{"relationships": [
  {{"source": "Adélaïde Fouque", "target": "Pierre Rougon", "type": "parent_child", "directed": true, "passage": "Pierre, le fils légitime", "confidence": 1.0}},
  {{"source": "Aristide Rougon", "target": "Eugène Rougon", "type": "sibling", "directed": false, "passage": "ses frères Eugène et Aristide", "confidence": 0.9}}
]}}
Each entry MUST have "source" (string), "target" (string), "type" (string), "directed" (bool), "passage" (string), and "confidence" (float).

IMPORTANT:
- Use ONLY canonical names from the provided list
- If a character is referenced but not in the list, skip that relationship
- Include relationships that are stated, implied, or can be inferred from context
- Do NOT invent relationships not supported by the text
""".format(types=", ".join(RELATIONSHIP_TYPES))


def _call_llm(client: OpenAI, chapter_key: str, passages_str: str, char_list_str: str, retries: int = 2) -> list[dict]:
    """Single LLM call for relationship extraction with retry."""
    for attempt in range(retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Canonical character list:\n{char_list_str}\n\n"
                        f"Passages from chapter {chapter_key}:\n\n{passages_str}"
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
            parsed = parsed.get("relationships", [])
            if not isinstance(parsed, list):
                parsed = []
        if not isinstance(parsed, list):
            return []

        validated = []
        for item in parsed:
            if isinstance(item, dict) and "source" in item and "target" in item:
                item.setdefault("type", "unknown")
                item.setdefault("directed", False)
                item.setdefault("passage", "")
                item.setdefault("confidence", 0.5)
                validated.append(item)
        return validated

    return []


MAX_PASSAGES_PER_CHUNK = 50


def extract_relationships_from_passages(
    client: OpenAI,
    chapter_key: str,
    passages: list[dict],
    characters: list[dict],
    cache_dir: Path,
) -> list[dict]:
    """Extract structured relationships from passages of a single chapter, with caching."""
    cache_file = cache_dir / f"relationships_{chapter_key}.json"
    if cache_file.exists():
        print(f"  [{chapter_key}] Using cached result")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if not passages:
        print(f"  [{chapter_key}] No passages, skipping")
        return []

    char_names = [c["canonical_name"] for c in characters]
    char_list_str = "\n".join(f"- {name}" for name in char_names)

    # Split into chunks if too many passages
    if len(passages) > MAX_PASSAGES_PER_CHUNK:
        chunks = [passages[i:i + MAX_PASSAGES_PER_CHUNK]
                   for i in range(0, len(passages), MAX_PASSAGES_PER_CHUNK)]
        print(f"  [{chapter_key}] Splitting {len(passages)} passages into {len(chunks)} chunks")
        parsed = []
        for ci, chunk in enumerate(chunks):
            passages_str = "\n\n".join(
                f"Passage {i+1} (hint: {p.get('relationship_hint', 'unknown')}): \"{p['text']}\""
                for i, p in enumerate(chunk)
            )
            print(f"  [{chapter_key}] Calling GPT-4o (chunk {ci+1}/{len(chunks)}, {len(chunk)} passages)...")
            parsed.extend(_call_llm(client, f"{chapter_key} (chunk {ci+1})", passages_str, char_list_str))
    else:
        passages_str = "\n\n".join(
            f"Passage {i+1} (hint: {p.get('relationship_hint', 'unknown')}): \"{p['text']}\""
            for i, p in enumerate(passages)
        )
        print(f"  [{chapter_key}] Calling GPT-4o ({len(passages)} passages)...")
        parsed = _call_llm(client, chapter_key, passages_str, char_list_str)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed


def normalize_and_dedup(all_rels: list[dict], characters: list[dict]) -> list[dict]:
    """Normalize edge direction and deduplicate relationships."""
    canonical_names = {c["canonical_name"] for c in characters}

    merged: dict[tuple, dict] = {}
    for rel in all_rels:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rtype = rel.get("type", "unknown")

        # Skip if characters not in canonical list
        if src not in canonical_names or tgt not in canonical_names:
            continue
        if src == tgt:
            continue

        # Normalize direction
        directed = rtype in DIRECTED_TYPES
        if not directed:
            # Symmetric: alphabetical order
            if src > tgt:
                src, tgt = tgt, src

        key = (src, tgt, rtype)
        if key in merged:
            existing = merged[key]
            # Accumulate passages
            passage = rel.get("passage", "")
            if passage and passage not in existing["passages"]:
                existing["passages"].append(passage)
            existing["weight"] += 1
            # Keep highest confidence
            existing["confidence"] = max(
                existing["confidence"], rel.get("confidence", 0.5)
            )
        else:
            merged[key] = {
                "source": src,
                "target": tgt,
                "type": rtype,
                "directed": directed,
                "passages": [rel.get("passage", "")],
                "weight": 1,
                "confidence": rel.get("confidence", 0.5),
            }

    return list(merged.values())


def main(
    passages_path: str,
    characters_path: str,
    output_path: str,
    cache_dir: str = "data/cache",
) -> list[dict]:
    all_passages = json.loads(Path(passages_path).read_text(encoding="utf-8"))
    characters = json.loads(Path(characters_path).read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = Path(cache_dir)

    all_rels = []
    for key, passages in tqdm(all_passages.items(), desc="Extracting relationships", unit="ch"):
        rels = extract_relationships_from_passages(client, key, passages, characters, cache)
        all_rels.extend(rels)
        tqdm.write(f"  [{key}] Found {len(rels)} relationships")

    print(f"Total raw relationships: {len(all_rels)}")
    relationships = normalize_and_dedup(all_rels, characters)
    print(f"After dedup: {len(relationships)} unique relationships")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(relationships, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in relationships:
        arrow = "→" if r["directed"] else "—"
        print(f"  {r['source']} {arrow} {r['target']} [{r['type']}] (x{r['weight']}, conf={r['confidence']})")

    return relationships


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract relationships from passages using GPT-4o")
    parser.add_argument("--passages", default="data/passages.json", help="Passages JSON")
    parser.add_argument("--characters", default="data/characters.json", help="Canonical characters JSON")
    parser.add_argument("--output", default="data/relationships.json", help="Output relationships JSON")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    args = parser.parse_args()
    main(args.passages, args.characters, args.output, args.cache_dir)
