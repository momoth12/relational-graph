# Relational Graph Extraction — La Fortune des Rougon

Pipeline to extract characters and their relationships from Émile Zola's novel using LLMs (GPT-4o for extraction, o3-mini for reasoning), then build and visualize a relational graph with NetworkX + Graphviz.

## Project Structure

```
relational-graph/
├── La_Fortune_des_Rougon.txt          # Source text (~3580 lines, UTF-8)
├── .env                                # OpenAI API key (OPENAI_API_KEY=sk-...)
├── requirements.txt                    # Python dependencies
├── src/                                # CLI modules (pipeline steps)
│   ├── __init__.py
│   ├── split_chapters.py              # Step 1: Split text into chapters
│   ├── extract_characters.py          # Step 2: Extract characters per chapter (LLM)
│   ├── merge_characters.py            # Step 3: Deduplicate into canonical list (LLM)
│   ├── extract_passages.py            # Step 4: Extract relationship passages (LLM)
│   ├── extract_relationships.py       # Step 5: Analyze passages into relationships (LLM)
│   ├── clean_relationships.py         # Step 5b: Validate & enrich relationships (o3-mini)
│   ├── build_graph.py                 # Step 6: Build NetworkX graph
│   └── visualize.py                   # Step 7: Render family tree images
├── data/                               # Intermediate outputs (auto-generated)
│   ├── cache/                          # Cached LLM responses (avoids redundant API calls)
│   ├── chapters.json                   # Step 1 output
│   ├── raw_characters.json             # Step 2 output
│   ├── characters.json                 # Step 3 output (canonical character sheets)
│   ├── passages.json                   # Step 4 output
│   ├── relationships.json              # Step 5 output (raw)
│   └── relationships_clean.json        # Step 5b output (cleaned & enriched)
├── output/
│   ├── graph.graphml                   # Step 6 output
│   ├── family_tree_all.png             # Step 7: Combined family tree
│   ├── family_tree_rougon.png          # Step 7: Rougon branch
│   ├── family_tree_macquart.png        # Step 7: Macquart branch
│   └── family_tree_mouret.png          # Step 7: Mouret branch
└── notebook.ipynb                      # Orchestration notebook (Colab-ready)
```

---

## Pipeline Steps

### Step 1 — Split Text into Chapters

**File:** `src/split_chapters.py`

**What it does:** Reads `La_Fortune_des_Rougon.txt` and splits it into 8 sections: the Préface + Chapters I–VII. Uses regex to detect chapter headings (roman numerals on their own line). Stops at the "FIN" marker.

**Input:** `La_Fortune_des_Rougon.txt`
**Output:** `data/chapters.json` — A JSON object mapping section keys to their text content:
```json
{
  "preface": "Je veux expliquer comment une famille...",
  "1": "Lorsqu'on sort de Plassans par la porte de Rome...",
  "2": "Plassans est une sous-préfecture...",
  ...
  "7": "Ce fut seulement le dimanche..."
}
```

**CLI:**
```bash
python -m src.split_chapters --input La_Fortune_des_Rougon.txt --output data/chapters.json
```


---

### Step 2 — Extract Characters from Each Chapter

**File:** `src/extract_characters.py`

**What it does:** Sends each chapter to GPT-4o with a structured prompt asking it to identify all named characters. For each character, the LLM returns the name, aliases (other ways the character is referred to), and a brief description. Responses are cached per chapter in `data/cache/` so re-running doesn't call the API again.

**Input:** `data/chapters.json`
**Output:** `data/raw_characters.json` — A JSON object mapping chapter keys to arrays of character objects:
```json
{
  "1": [
    {"name": "Silvère Mouret", "aliases": ["Silvère"], "description": "Young worker and republican insurgent"},
    {"name": "Miette", "aliases": ["Marie"], "description": "Young girl, Silvère's lover"}
  ],
  "2": [ ... ]
}
```

**CLI:**
```bash
python -m src.extract_characters --input data/chapters.json --output data/raw_characters.json
```

**API calls:** 1 per chapter (8 total). Cached after first run.

---

### Step 3 — Merge into Canonical Character List

**File:** `src/merge_characters.py`

**What it does:** Takes the per-chapter raw characters and merges them into a single deduplicated list. Each character gets a **canonical name** (their most complete full name, e.g., "Pierre Rougon" not just "Pierre"). The LLM resolves ambiguities (e.g., "le docteur" → "Pascal Rougon"). Each character is tagged with:
- `canonical_name` — unique identifier used everywhere (graph nodes, JSON keys)
- `aliases` — all other names/references for this character
- `description` — merged description from all chapters
- `family_branch` — "Rougon", "Macquart", "Mouret", or "other"
- `chapters` — which chapters they appear in

**Input:** `data/raw_characters.json`
**Output:** `data/characters.json` — A JSON array of canonical character sheets:
```json
[
  {
    "canonical_name": "Pierre Rougon",
    "aliases": ["Rougon", "Pierre", "M. Rougon"],
    "description": "Ambitious merchant, head of the Rougon family",
    "family_branch": "Rougon",
    "chapters": ["1", "3", "5", "6", "7"]
  }
]
```

**CLI:**
```bash
python -m src.merge_characters --input data/raw_characters.json --output data/characters.json
```

**API calls:** 1 (sends all raw characters at once). Cached after first run.

---

### Step 4 — Extract Relationship Passages

**File:** `src/extract_passages.py`

**What it does:** Sends each chapter + the canonical character list to GPT-4o and asks it to identify and extract **only the text passages** where a relationship between characters is being described, implied, or revealed. This filters the full novel text down to relationship-relevant excerpts, providing focused input for the next step.

A "relationship passage" includes:
- Family ties being stated ("son fils Pierre", "sa mère Adélaïde")
- Romantic or marital bonds ("sa femme Félicité", "Silvère aimait Miette")
- Political alliances or oppositions
- Friendships, enmities, professional ties
- Any interaction that reveals the nature of a bond between characters

**Input:** `data/chapters.json` + `data/characters.json`
**Output:** `data/passages.json` — A JSON object mapping chapter keys to arrays of passage objects:
```json
{
  "1": [
    {"text": "Pierre, le fils légitime d'Adélaïde, grandit dans la haine", "characters": ["Adélaïde Fouque", "Pierre Rougon"], "relationship_hint": "parent-child"},
    {"text": "Félicité, sa femme, une petite femme sèche", "characters": ["Pierre Rougon", "Félicité Rougon"], "relationship_hint": "spouses"}
  ]
}
```

**CLI:**
```bash
python -m src.extract_passages --chapters data/chapters.json --characters data/characters.json --output data/passages.json
```

**API calls:** 1 per chapter (8 total). Cached after first run.

---

### Step 5 — Analyze Passages into Structured Relationships

**File:** `src/extract_relationships.py`

**What it does:** Takes the extracted passages (from Step 4) and the canonical character list, sends them to GPT-4o to produce structured relationship data. Because the input is pre-filtered to relationship-relevant text only, this step is more focused and accurate than analyzing full chapters directly.
- `source` / `target` — canonical names (from the provided list)
- `type` — relationship type (parent_child, spouse, sibling, lover, friend, enemy, political_ally, etc.)
- `directed` — whether the relationship is asymmetric
- `passage` — a short quote from the text as evidence
- `confidence` — 0.0 to 1.0

**Edge normalization rules (built-in):**
- **Asymmetric** (parent_child, grandparent_grandchild, uncle_nephew, employer_employee): source = the "higher" role (parent, grandparent, etc.)
- **Symmetric** (spouse, sibling, lover, friend, enemy, etc.): alphabetical order on canonical names

After extraction, relationships are **deduplicated** across chapters: same (source, target, type) = one edge. Multiple passages are accumulated, and a `weight` (frequency count) is assigned.

**Input:** `data/passages.json` + `data/characters.json`
**Output:** `data/relationships.json` — A JSON array of unique relationships:
```json
[
  {
    "source": "Adélaïde Fouque",
    "target": "Pierre Rougon",
    "type": "parent_child",
    "directed": true,
    "passages": ["Pierre, le fils légitime", "sa mère Adélaïde"],
    "weight": 3,
    "confidence": 1.0
  }
]
```

**CLI:**
```bash
python -m src.extract_relationships --passages data/passages.json --characters data/characters.json --output data/relationships.json
```

**API calls:** 1 per chapter (8 total). Cached after first run.

---

### Step 5b — Validate & Enrich Relationships

**File:** `src/clean_relationships.py`

**What it does:** Uses **o3-mini** (reasoning model) to validate, correct, and complete the extracted relationships by cross-referencing them with character descriptions. This step leverages a reasoning model rather than a standard completion model because it requires logical deduction (e.g., "if A is the son of B, then B→A is parent_child, not A→B").

**Validation pipeline (4 sub-steps):**
1. **Normalize types** — Maps non-canonical types to standard ones (`in-laws` → `in_law`, `cousins` → `cousin`, etc.)
2. **LLM validation (o3-mini)** — Sends all relationships + character descriptions to o3-mini, which:
   - Fixes wrong relationship types (e.g., parent_child → spouse)
   - Corrects reversed directions on directed edges (parent_child, grandparent_grandchild, uncle_nephew)
   - Removes false positives (e.g., a parent-child pair mislabeled as siblings)
   - Infers missing relationships from character descriptions (e.g., shared parents → sibling edges)
3. **Normalize symmetric edges** — Ensures alphabetical source < target for undirected types
4. **Deduplicate** — Merges duplicate edges, keeps highest confidence, accumulates passages

**Why o3-mini instead of GPT-4o?** GPT-4o consistently fails at direction inference — it confuses who is the parent vs child based on dialogue passages (e.g., "Mon père a sauvé la ville, dit Aristide" → GPT-4o puts Aristide as the parent). o3-mini's chain-of-thought reasoning handles this correctly.

**Input:** `data/relationships.json` + `data/characters.json`
**Output:** `data/relationships_clean.json`

**CLI:**
```bash
python -m src.clean_relationships --input data/relationships.json --characters data/characters.json --output data/relationships_clean.json
```

**API calls:** 1 (o3-mini). Cached after first run.

---

### Step 6 — Build Graph

**File:** `src/build_graph.py`

**What it does:** Constructs a NetworkX `DiGraph` from the canonical characters and relationships.

- **Nodes** = canonical characters, with attributes: aliases, description, family_branch, chapters, uncertain (bool)
- **Edges** = relationships, with attributes: relationship_type, directed, weight, confidence, passages
- **Placeholder nodes ("faux nœuds"):** If a relationship references a character not in the canonical list, a placeholder node is created with `uncertain=True`
- **Symmetric edges:** For undirected relationships, edges are added in both directions
- **Fork support:** The `fork_graph()` function can duplicate the graph with alternative edges for ambiguous relationships

**Input:** `data/characters.json` + `data/relationships_clean.json`
**Output:** `output/graph.graphml` — GraphML file (importable in Gephi, Cytoscape, etc.)

**CLI:**
```bash
python -m src.build_graph --characters data/characters.json --relationships data/relationships_clean.json --output output/graph.graphml
```

**No API call.** Pure graph construction.

---

### Step 7 — Visualize Family Trees

**File:** `src/visualize.py`

**What it does:** Generates genealogical family tree images using **pygraphviz AGraph** with Graphviz rank constraints. Produces one combined tree + one per family branch (Rougon, Macquart, Mouret).

**Key features:**
- **Generation-based layout:** Assigns generation numbers via BFS from root ancestors (Adélaïde Fouque = Gen 0). Uses `rank="same"` Graphviz subgraphs to force horizontal alignment of each generation.
- **Strict filtering:** Only shows `parent_child`, `spouse`, and `sibling` edges. Uncle/nephew, cousin, in-law edges are excluded from tree views (redundant with parent_child chains, cause visual noise).
- **Horizontal constraints:** Spouse and sibling edges use `constraint=false` so they stay horizontal without pulling nodes to different vertical ranks.
- **Orthogonal edges:** Uses `splines=ortho` for clean right-angle connections.
- **Direct Graphviz rendering:** Uses `pygraphviz.AGraph.draw()` for full control over rank/constraint attributes.

**Visual encoding:**
| Element | Encoding |
|---|---|
| Node fill | Family branch: Rougon = light blue, Macquart = light red, Mouret = light green, other = light gray |
| Node border | Family branch color (saturated) |
| Node shape | Rounded box |
| `parent_child` edge | Solid black arrow, vertical (downward) |
| `spouse` edge | Bold pink line, no arrow, horizontal |
| `sibling` edge | Dashed blue line, no arrow, horizontal |

**Input:** `output/graph.graphml`
**Output:** `output/family_tree_all.png`, `output/family_tree_rougon.png`, `output/family_tree_macquart.png`, `output/family_tree_mouret.png`

**CLI:**
```bash
python -m src.visualize --input output/graph.graphml --output output
```

**No API call.** Pure rendering.

---

## Orchestration Notebook

**File:** `notebook.ipynb`

The notebook calls each step sequentially via Python imports (not shell commands), so all intermediate data is available in-memory for inspection between steps. Each step also writes to `data/` so it can be re-run independently.

The notebook is **Colab-ready**: uncomment the first cell to install dependencies on Colab.

---

## Caching

All LLM responses are cached in `data/cache/`. To force a re-extraction:
- Delete the specific cache file (e.g., `data/cache/characters_1.json` for chapter 1)
- Or delete the entire `data/cache/` directory to re-run everything

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | GPT-4o & o3-mini & o3-mini API calls |
| `python-dotenv` | Load API key from `.env` |
| `networkx` | Graph data structure & GraphML export |
| `pygraphviz` | Graphviz-based family tree rendering |
| `tqdm` | Progress bars for LLM extraction steps |

**Note:** On Windows, `pygraphviz` requires Graphviz C libraries. Easiest install via conda:
```bash
conda install -c conda-forge graphviz pygraphviz
```
