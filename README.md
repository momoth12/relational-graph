# Relational Graph Extraction

Extract characters and relationships from a novel using LLMs (GPT-4o + o3-mini), then build and visualize a family tree with NetworkX and Graphviz.

The pipeline is book-agnostic: edit `book_config.json` to target a different novel.

## Quick start

```bash
# 1. Create .env with your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 2. Install dependencies
pip install -r requirements.txt
conda install -c conda-forge graphviz pygraphviz   # pygraphviz needs C libs

# 3. Run the pipeline
python -m src.split_chapters
python -m src.extract_characters   --book-config book_config.json
python -m src.merge_characters     --book-config book_config.json
python -m src.extract_passages     --book-config book_config.json
python -m src.extract_relationships --book-config book_config.json
python -m src.clean_relationships
python -m src.build_graph --characters data/characters.json --relationships data/relationships_clean.json --output output/graph.graphml
python -m src.visualize --book-config book_config.json
```

Or run everything interactively via `notebook.ipynb`.

## Pipeline

| Step | Module | Model | Description |
|------|--------|-------|-------------|
| 1 | `split_chapters.py` | — | Split source text into chapters |
| 2 | `extract_characters.py` | GPT-4o | Extract named characters per chapter |
| 3 | `merge_characters.py` | GPT-4o | Deduplicate into canonical character list |
| 4 | `extract_passages.py` | GPT-4o | Extract relationship-revealing passages |
| 5 | `extract_relationships.py` | GPT-4o | Structure passages into typed relationships |
| 5b | `clean_relationships.py` | o3-mini | Validate directions, fix errors, infer missing edges |
| 6 | `build_graph.py` | — | Build NetworkX DiGraph, export GraphML |
| 7 | `visualize.py` | — | Render family tree PNGs via Graphviz |

All LLM responses are cached in `data/cache/` — delete a cache file to re-run that step.

## Configuration

`book_config.json` defines the target book:

```json
{
  "title": "La Fortune des Rougon",
  "author": "Émile Zola",
  "language": "French",
  "period": "19th-century",
  "groups": ["Rougon", "Macquart", "Mouret"]
}
```

## Output

- `output/graph.graphml` — full relational graph (importable in Gephi, Cytoscape)
- `output/family_tree_*.png` — family tree visualizations (one combined + one per group)

## Dependencies

`openai`, `python-dotenv`, `networkx`, `pygraphviz`, `tqdm`
