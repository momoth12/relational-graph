"""Clean, normalize and enrich the extracted relationships.

Step 5b — runs between extract_relationships and build_graph.
Uses GPT-4o to validate and correct relationships by cross-referencing
with character descriptions from characters.json.

Pipeline:
1. Normalize non-canonical relationship types (deterministic)
2. LLM validation: correct errors, fix directions, infer missing (GPT-4o)
3. Normalize symmetric edge ordering (deterministic)
4. Deduplicate (deterministic)
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Type normalization map ───────────────────────────────────────────────────
TYPE_MAP = {
    # Canonical types (pass through)
    "parent_child": "parent_child",
    "spouse": "spouse",
    "sibling": "sibling",
    "lover": "lover",
    "grandparent_grandchild": "grandparent_grandchild",
    "uncle_nephew": "uncle_nephew",
    "friend": "friend",
    "enemy": "enemy",
    "political_ally": "political_ally",
    "political_opponent": "political_opponent",
    "employer_employee": "employer_employee",
    "neighbor": "neighbor",
    "acquaintance": "acquaintance",
    # Non-canonical → canonical
    "uncle_niece": "uncle_nephew",
    "cousin": "cousin",
    "cousins": "cousin",
    "in-law": "in_law",
    "in-laws": "in_law",
    "in_law": "in_law",
    "in_laws": "in_law",
    "political_enemy": "political_opponent",
    "family": "family",
    "family_ties": "family",
    "fellow prisoners": "acquaintance",
}

DIRECTED_TYPES = {"parent_child", "grandparent_grandchild", "uncle_nephew", "employer_employee"}

CACHE_PATH = Path("data/cache/clean_validation.json")

SYSTEM_PROMPT = """\
You are a literary analyst specializing in extracting genealogical data from novels.

You will receive:
1. A list of canonical characters with their descriptions (which contain family information)
2. A list of extracted relationships between these characters

Your task is to produce a CORRECTED and COMPLETE list of relationships by cross-referencing
the character descriptions with the extracted relationships.

CRITICAL METHODOLOGY — Follow these steps IN ORDER:

STEP 1: Build a mental family model.
Read ALL character descriptions first. Extract every family fact:
- "son of X and Y" → X is parent, Y is parent, the described character is the child
- "daughter of X" → X is parent
- "mother/father of X" → described character is parent, X is child
- "married to X" / "wife/husband of X" → spouse
- "brother/sister of X" → sibling
- "uncle/aunt of X" → uncle_nephew (uncle=source, nephew=target)
- "grandfather/grandmother of X" → grandparent_grandchild

STEP 2: Validate each extracted relationship against your family model.
A) Fix wrong types: if the descriptions contradict the type, correct it.
B) Fix reversed directions: for directed types, source MUST be the ascendant/superior.
   - parent_child: source=parent, target=child
   - grandparent_grandchild: source=grandparent, target=grandchild
   - uncle_nephew: source=uncle/aunt, target=nephew/niece
   - employer_employee: source=employer, target=employee
C) Remove false positives: delete edges that contradict the descriptions.
   A parent and their child can NEVER be siblings. If A→B is parent_child, delete any sibling edge between A and B.
D) Remove duplicate directed edges: for each directed pair, there must be exactly ONE edge with the correct direction.

STEP 3: Add missing relationships from the descriptions.
- If a description says "son of X and Y" but only one parent_child edge exists, add the other.
- If two characters share a parent, add a sibling edge between them.
- Spouse edges mentioned in descriptions but missing should be added.

STEP 4: Preserve non-genealogical relationships as-is (political_ally, enemy, friend, etc.).

For symmetric types (spouse, sibling, lover, friend, enemy, etc.): source = alphabetically first name.

Output format: a JSON object with key "relationships" containing an array.
Each relationship: {source, target, type, directed (bool), confidence (0.0-1.0), passages (array of strings)}.
For corrected or new edges: passages = ["[LLM validation]"], confidence = 1.0.
For unchanged edges: preserve original passages and confidence.

Return ALL relationships (corrected + new + unchanged non-genealogical).
Use ONLY canonical names from the character list.
"""


def _make_key(src: str, tgt: str, rtype: str) -> tuple:
    """Create a dedup key. Symmetric types use alphabetical order."""
    if rtype not in DIRECTED_TYPES:
        if src > tgt:
            src, tgt = tgt, src
    return (src, tgt, rtype)


def normalize_types(rels: list[dict]) -> list[dict]:
    """Map all relationship types to canonical forms."""
    for r in rels:
        original = r["type"]
        r["type"] = TYPE_MAP.get(original, original)
    return rels


def llm_validate(rels: list[dict], characters: list[dict]) -> list[dict]:
    """Use GPT-4o to validate, correct and complete relationships."""
    # Check cache first
    if CACHE_PATH.exists():
        print("  Using cached LLM validation result")
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return cached

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build the user message
    char_section = "CANONICAL CHARACTERS:\n"
    for c in characters:
        aliases = ", ".join(c.get("aliases", []))
        char_section += f"- {c['canonical_name']} (branch: {c.get('family_branch', 'other')}): {c.get('description', 'N/A')}"
        if aliases:
            char_section += f" [aliases: {aliases}]"
        char_section += "\n"

    # Strip passages from input to save tokens (o3-mini uses reasoning tokens)
    rels_compact = []
    for r in rels:
        rels_compact.append({
            "source": r["source"],
            "target": r["target"],
            "type": r["type"],
            "directed": r.get("directed", False),
        })

    rels_section = "\nEXTRACTED RELATIONSHIPS:\n"
    rels_section += json.dumps(rels_compact, ensure_ascii=False, indent=2)

    user_msg = char_section + rels_section

    print(f"  Sending {len(rels)} relationships + {len(characters)} characters to o3-mini...")

    response = client.chat.completions.create(
        model="o3-mini",
        max_completion_tokens=32768,
        response_format={"type": "json_object"},
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    result = data.get("relationships", [])

    # Re-attach original passages where possible
    orig_map = {}
    for r in rels:
        key = _make_key(r["source"], r["target"], r["type"])
        orig_map[key] = r

    # Validate structure
    validated = []
    for r in result:
        if not isinstance(r, dict) or "source" not in r or "target" not in r:
            continue
        r.setdefault("type", "unknown")
        r.setdefault("directed", r["type"] in DIRECTED_TYPES)
        # Restore original passages if this edge existed before
        key = _make_key(r["source"], r["target"], r["type"])
        if key in orig_map:
            orig = orig_map[key]
            r.setdefault("confidence", orig.get("confidence", 0.8))
            r.setdefault("passages", orig.get("passages", ["[LLM validation]"]))
        else:
            r.setdefault("confidence", 1.0)
            r.setdefault("passages", ["[LLM validation]"])
        r.setdefault("weight", len(r["passages"]))
        validated.append(r)

    print(f"  LLM returned {len(validated)} relationships")

    # Cache the result
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")

    return validated


def normalize_symmetric(rels: list[dict]) -> list[dict]:
    """Ensure symmetric relations use alphabetical source < target."""
    for r in rels:
        if r["type"] not in DIRECTED_TYPES:
            r["directed"] = False
            if r["source"] > r["target"]:
                r["source"], r["target"] = r["target"], r["source"]
    return rels


def dedup(rels: list[dict]) -> list[dict]:
    """Merge duplicate edges, keeping the one with highest confidence and merging passages."""
    merged: dict[tuple, dict] = {}
    for r in rels:
        key = _make_key(r["source"], r["target"], r["type"])
        if key in merged:
            existing = merged[key]
            # Merge passages
            existing_passages = set(existing.get("passages", []))
            new_passages = set(r.get("passages", []))
            existing["passages"] = list(existing_passages | new_passages)
            existing["weight"] = len(existing["passages"])
            existing["confidence"] = max(existing.get("confidence", 0), r.get("confidence", 0))
        else:
            merged[key] = r.copy()
    return list(merged.values())


def clean(rels: list[dict], characters: list[dict]) -> list[dict]:
    """Run the full cleaning pipeline."""
    print("Step 1: Normalize types")
    rels = normalize_types(rels)

    print("Step 2: LLM validation (correct, fix directions, infer missing)")
    rels = llm_validate(rels, characters)

    print("Step 3: Normalize symmetric edges")
    rels = normalize_symmetric(rels)

    print("Step 4: Deduplicate")
    before = len(rels)
    rels = dedup(rels)
    print(f"  {before} → {len(rels)} edges")

    # Sort for readability
    rels.sort(key=lambda r: (r["type"], r["source"], r["target"]))

    return rels


def main(
    relationships_path: str,
    characters_path: str,
    output_path: str,
) -> list[dict]:
    rels = json.loads(Path(relationships_path).read_text(encoding="utf-8"))
    characters = json.loads(Path(characters_path).read_text(encoding="utf-8"))

    print(f"Input: {len(rels)} relationships")
    rels = clean(rels, characters)
    print(f"Output: {len(rels)} relationships")

    # Summary by type
    types: dict[str, int] = {}
    for r in rels:
        types[r["type"]] = types.get(r["type"], 0) + 1
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rels, ensure_ascii=False, indent=2), encoding="utf-8")
    return rels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and enrich relationships")
    parser.add_argument("--input", default="data/relationships.json", help="Raw relationships JSON")
    parser.add_argument("--characters", default="data/characters.json", help="Canonical characters JSON")
    parser.add_argument("--output", default="data/relationships_clean.json", help="Cleaned relationships JSON")
    args = parser.parse_args()
    main(args.input, args.characters, args.output)