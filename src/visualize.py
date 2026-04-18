"""Visualize family trees as PNG images via pygraphviz / Graphviz."""


import argparse
import json
from collections import deque
from pathlib import Path

import networkx as nx
import pygraphviz as pgv

# (border, fill) color pairs assigned to groups in alphabetical order
_GROUP_PALETTE = [
    ("#2E6BA6", "#CADCEF"),
    ("#B8433A", "#F2D1CE"),
    ("#3A8C47", "#D0ECCE"),
    ("#C78A2E", "#F5E4C8"),
    ("#7B4FB0", "#E0D4F0"),
]
_DEFAULT_COLORS = ("#888888", "#F0F0F0")

# Only these types appear in family tree views
TREE_TYPES = {"parent_child", "spouse", "sibling"}

# Edge visual properties per type
EDGE_STYLES = {
    "parent_child": {"color": "#333", "penwidth": "2.5", "style": "solid", "arrowhead": "normal"},
    "spouse":       {"color": "#B8433A", "penwidth": "2", "style": "bold", "arrowhead": "none", "constraint": "false"},
    "sibling":      {"color": "#2E6BA6", "penwidth": "1.5", "style": "dashed", "arrowhead": "none", "constraint": "false"},
}

def _assign_generations(G: nx.DiGraph) -> dict[str, int]:
    """BFS from roots (gen 0) along parent_child edges."""
    # parent->child adjacency
    children_of: dict[str, list[str]] = {}
    parents_of: dict[str, list[str]] = {}
    for u, v, d in G.edges(data=True):
        if d.get("relationship_type") == "parent_child":
            children_of.setdefault(u, []).append(v)
            parents_of.setdefault(v, []).append(u)

    all_parents = set(children_of.keys())
    all_children = set(parents_of.keys())
    all_pc_nodes = all_parents | all_children
    roots = all_parents - all_children
    if not roots:
        roots = all_pc_nodes

    gen: dict[str, int] = {}
    queue = deque()
    for r in roots:
        gen[r] = 0
        queue.append(r)

    while queue:
        node = queue.popleft()
        for child in children_of.get(node, []):
            if child not in gen:
                gen[child] = gen[node] + 1
                queue.append(child)

    # spouses share the same generation
    for u, v, d in G.edges(data=True):
        if d.get("relationship_type") == "spouse":
            if u in gen and v not in gen:
                gen[v] = gen[u]
            elif v in gen and u not in gen:
                gen[u] = gen[v]

    for n in G.nodes():
        if n not in gen:
            gen[n] = -1

    return gen


def _filter_tree_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Keep only family edges and drop isolates."""
    H = G.copy()
    to_remove = [(u, v) for u, v, d in H.edges(data=True)
                 if d.get("relationship_type", "") not in TREE_TYPES]
    H.remove_edges_from(to_remove)
    H.remove_nodes_from(list(nx.isolates(H)))
    return H


def _get_group_subgraph(G: nx.DiGraph, group: str) -> nx.DiGraph:
    """Subgraph of a group and its direct connections."""
    group_nodes = {n for n, d in G.nodes(data=True) if d.get("group") == group}
    connected = set()
    for n in group_nodes:
        for _, v, _ in G.edges(n, data=True):
            connected.add(v)
        for u, _, _ in G.in_edges(n, data=True):
            connected.add(u)
    return G.subgraph(group_nodes | connected).copy()


def _build_group_colors(G: nx.DiGraph) -> tuple[dict, dict]:
    """Map each group to a border/fill color pair."""
    groups = sorted({d.get("group", "other") for _, d in G.nodes(data=True)} - {"other"})
    border = {}
    fill = {}
    for i, g in enumerate(groups):
        b, f = _GROUP_PALETTE[i % len(_GROUP_PALETTE)]
        border[g] = b
        fill[g] = f
    border["other"] = _DEFAULT_COLORS[0]
    fill["other"] = _DEFAULT_COLORS[1]
    return border, fill


def _build_agraph(G: nx.DiGraph, gen: dict[str, int], title: str,
                  border_colors: dict | None = None,
                  fill_colors: dict | None = None) -> pgv.AGraph:
    """Convert a NetworkX DiGraph into a pygraphviz AGraph with generation ranks."""
    A = pgv.AGraph(directed=True, strict=False)
    A.graph_attr.update(
        rankdir="TB",
        ranksep="1.0",
        nodesep="0.6",
        splines="ortho",
        label=title,
        labelloc="t",
        fontsize="20",
        fontname="Helvetica",
        bgcolor="white",
        pad="0.5",
    )
    A.node_attr.update(
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="11",
        margin="0.15,0.08",
    )
    A.edge_attr.update(fontname="Helvetica", fontsize="9")

    for n in G.nodes():
        data = G.nodes[n]
        group = data.get("group", "other")
        fillcolor = (fill_colors or {}).get(group, _DEFAULT_COLORS[1])
        bordercolor = (border_colors or {}).get(group, _DEFAULT_COLORS[0])
        A.add_node(n, fillcolor=fillcolor, color=bordercolor, penwidth="2.0")

    seen_undirected = set()
    for u, v, d in G.edges(data=True):
        rtype = d.get("relationship_type", "")
        is_directed = d.get("directed", False)

        if not is_directed:
            key = tuple(sorted([u, v]))
            if key in seen_undirected:
                continue
            seen_undirected.add(key)

        edge_attrs = dict(EDGE_STYLES.get(rtype, {}))

        if not is_directed:
            edge_attrs["dir"] = "none"

        A.add_edge(u, v, **edge_attrs)

    gen_groups: dict[int, list[str]] = {}
    for n in G.nodes():
        g = gen.get(n, -1)
        if g >= 0:
            gen_groups.setdefault(g, []).append(n)

    for g in sorted(gen_groups.keys()):
        nodes = gen_groups[g]
        subg = A.add_subgraph(nodes, name=f"gen_{g}")
        subg.graph_attr["rank"] = "same"

    # Legend
    legend_nodes = []
    idx = 0

    groups_present = sorted(
        {G.nodes[n].get("group", "other") for n in G.nodes()}
    )
    for grp in groups_present:
        nid = f"_legend_grp_{idx}"
        fc = (fill_colors or {}).get(grp, _DEFAULT_COLORS[1])
        bc = (border_colors or {}).get(grp, _DEFAULT_COLORS[0])
        A.add_node(nid, label=grp, shape="box", style="filled,rounded",
                   fillcolor=fc, color=bc, penwidth="2.0",
                   fontname="Helvetica", fontsize="10", width="1.2", height="0.3")
        legend_nodes.append(nid)
        idx += 1

    edge_labels = {
        "parent_child": "Parent -> Child",
        "spouse": "Spouse",
        "sibling": "Sibling",
    }
    for rtype, label in edge_labels.items():
        src_id = f"_legend_esrc_{idx}"
        tgt_id = f"_legend_etgt_{idx}"
        A.add_node(src_id, label="", shape="point", width="0.01", height="0.01")
        A.add_node(tgt_id, label=label, shape="plaintext",
                   fontname="Helvetica", fontsize="10")
        style_attrs = dict(EDGE_STYLES.get(rtype, {}))
        if rtype in ("spouse", "sibling"):
            style_attrs["dir"] = "none"
        style_attrs["constraint"] = "false"
        A.add_edge(src_id, tgt_id, **style_attrs)
        legend_nodes.extend([src_id, tgt_id])
        idx += 1

    legend_sub = A.add_subgraph(legend_nodes, name="cluster_legend")
    legend_sub.graph_attr.update(
        label="Legend", fontname="Helvetica", fontsize="12",
        style="rounded", color="#AAAAAA", bgcolor="#F9F9F9",
        rank="sink",
    )

    return A


def _render(A: pgv.AGraph, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.draw(str(out), prog="dot", format="png")
    print(f"  Saved: {output_path}")


def visualize_family_trees(G: nx.DiGraph, output_dir: str,
                           book_title: str = "Novel") -> None:
    """One tree per group + one combined."""
    H = _filter_tree_edges(G)
    print(f"Tree subgraph: {len(H.nodes())} nodes, {len(H.edges())} edges")

    gen = _assign_generations(G)
    border_colors, fill_colors = _build_group_colors(G)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    A = _build_agraph(H, gen, f"Family tree \u2014 {book_title}", border_colors, fill_colors)
    _render(A, str(out / "family_tree_all.png"))

    groups = sorted({d.get("group", "other") for _, d in H.nodes(data=True)} - {"other"})
    for group in groups:
        sub = _get_group_subgraph(H, group)
        if len(sub.nodes()) > 1:
            A = _build_agraph(sub, gen, f"Family tree \u2014 {group}", border_colors, fill_colors)
            _render(A, str(out / f"family_tree_{group.lower()}.png"))


def _load_book_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(input_path: str, output_dir: str, prog: str = "dot",
         book_config_path: str = "book_config.json") -> None:
    G = nx.read_graphml(input_path)
    for node in G.nodes():
        data = G.nodes[node]
        if isinstance(data.get("uncertain"), str):
            data["uncertain"] = data["uncertain"].lower() == "true"
    for u, v in G.edges():
        data = G.edges[u, v]
        if isinstance(data.get("directed"), str):
            data["directed"] = data["directed"].lower() == "true"

    book_config = _load_book_config(book_config_path)
    visualize_family_trees(G, output_dir, book_title=book_config["title"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate family tree visualizations")
    parser.add_argument("--input", default="output/graph.graphml", help="Input GraphML path")
    parser.add_argument("--output", default="output", help="Output directory for tree images")
    parser.add_argument("--prog", default="dot", help="Graphviz layout program (unused, kept for compat)")
    parser.add_argument("--book-config", default="book_config.json", help="Book configuration JSON")
    args = parser.parse_args()
    main(args.input, args.output, args.prog, args.book_config)
