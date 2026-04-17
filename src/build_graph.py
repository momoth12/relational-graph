"""Step 6 — Build a NetworkX graph from characters and relationships."""


import argparse
import copy
import json
from pathlib import Path

import networkx as nx


def build_graph(characters: list[dict], relationships: list[dict]) -> nx.DiGraph:
    """Build a directed graph from characters and relationships."""
    G = nx.DiGraph()

    for char in characters:
        G.add_node(
            char["canonical_name"],
            aliases=", ".join(char.get("aliases", [])),
            description=char.get("description", ""),
            group=char.get("group", char.get("family_branch", "other")),
            chapters=", ".join(char.get("chapters", [])),
            uncertain=False,
        )

    for rel in relationships:
        src = rel["source"]
        tgt = rel["target"]

        for node in (src, tgt):
            if node not in G:
                G.add_node(
                    node,
                    aliases="",
                    description="Unknown character",
                    group="other",
                    chapters="",
                    uncertain=True,
                )

        G.add_edge(
            src,
            tgt,
            relationship_type=rel["type"],
            directed=rel.get("directed", False),
            weight=rel.get("weight", 1),
            confidence=rel.get("confidence", 0.5),
            passages="; ".join(rel.get("passages", [])),
        )

        if not rel.get("directed", False):
            G.add_edge(
                tgt,
                src,
                relationship_type=rel["type"],
                directed=False,
                weight=rel.get("weight", 1),
                confidence=rel.get("confidence", 0.5),
                passages="; ".join(rel.get("passages", [])),
            )

    return G


def fork_graph(
    G: nx.DiGraph, edge_key: tuple[str, str], alternatives: list[dict]
) -> list[nx.DiGraph]:
    """Fork a graph for an ambiguous relationship.

    Returns one copy per alternative interpretation."""
    variants = []
    src, tgt = edge_key
    for alt in alternatives:
        G_copy = copy.deepcopy(G)
        if G_copy.has_edge(src, tgt):
            G_copy.remove_edge(src, tgt)
        new_src = alt.get("source", src)
        new_tgt = alt.get("target", tgt)
        G_copy.add_edge(
            new_src,
            new_tgt,
            relationship_type=alt.get("type", "unknown"),
            directed=alt.get("directed", False),
            weight=alt.get("weight", 1),
            confidence=alt.get("confidence", 0.5),
            passages=alt.get("passages", ""),
        )
        variants.append(G_copy)
    return variants


def main(
    characters_path: str,
    relationships_path: str,
    output_path: str,
) -> nx.DiGraph:
    characters = json.loads(Path(characters_path).read_text(encoding="utf-8"))
    relationships = json.loads(Path(relationships_path).read_text(encoding="utf-8"))

    G = build_graph(characters, relationships)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(out))

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    uncertain = [n for n, d in G.nodes(data=True) if d.get("uncertain")]
    if uncertain:
        print(f"Uncertain/placeholder nodes: {uncertain}")

    low_conf = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("confidence", 1) < 0.7]
    if low_conf:
        print(f"Low-confidence edges ({len(low_conf)}):")
        for u, v, d in low_conf:
            print(f"  {u} → {v} [{d['relationship_type']}] conf={d['confidence']}")

    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NetworkX graph from characters + relationships")
    parser.add_argument("--characters", default="data/characters.json", help="Canonical characters JSON")
    parser.add_argument("--relationships", default="data/relationships.json", help="Relationships JSON")
    parser.add_argument("--output", default="data/graph.graphml", help="Output GraphML path")
    args = parser.parse_args()
    main(args.characters, args.relationships, args.output)
