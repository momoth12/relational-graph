"""Visualize family trees using pygraphviz AGraph with generation-based ranks.

Produces one tree per family branch (Rougon, Macquart, Mouret) + one combined,
filtering to core genealogical relationships (parent_child, spouse, sibling).
Uses Graphviz rank constraints to align generations horizontally.
"""

import argparse
from collections import deque
from pathlib import Path

import networkx as nx
import pygraphviz as pgv

# ─── Configuration ────────────────────────────────────────────────────────────

BRANCH_COLORS = {
    "Rougon": "#4A90D9",
    "Macquart": "#D94A4A",
    "Mouret": "#4AD94A",
    "other": "#C0C0C0",
}
BRANCH_FILLCOLORS = {
    "Rougon": "#DAE8F8",
    "Macquart": "#F8DADA",
    "Mouret": "#DAF8DA",
    "other": "#EEEEEE",
}

# Only these types appear in family tree views
TREE_TYPES = {"parent_child", "spouse", "sibling"}

# Edge visual properties per type
EDGE_STYLES = {
    "parent_child": dict(color="#333333", penwidth="2.5", style="solid", arrowhead="normal"),
    "spouse":       dict(color="#D94A9A", penwidth="2.0", style="bold",  arrowhead="none", constraint="false"),
    "sibling":      dict(color="#4A90D9", penwidth="1.5", style="dashed", arrowhead="none", constraint="false"),
}

# ─── Generation assignment ────────────────────────────────────────────────────

def _assign_generations(G: nx.DiGraph) -> dict[str, int]:
    """BFS from roots using parent_child edges to assign generation numbers.
    
    Gen 0 = nodes with no incoming parent_child edges (roots like Adélaïde).
    Only parent_child edges are traversed.
    """
    # Build a parent→child adjacency from the full graph
    children_of: dict[str, list[str]] = {}
    parents_of: dict[str, list[str]] = {}
    for u, v, d in G.edges(data=True):
        if d.get("relationship_type") == "parent_child":
            children_of.setdefault(u, []).append(v)
            parents_of.setdefault(v, []).append(u)

    # Find roots: nodes that appear as parents but not as children in parent_child edges
    all_parents = set(children_of.keys())
    all_children = set(parents_of.keys())
    all_pc_nodes = all_parents | all_children
    roots = all_parents - all_children
    if not roots:
        # Fallback: pick nodes with no parent_child predecessors
        roots = all_pc_nodes

    # BFS to assign generations
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

    # Assign spouse same generation as their partner
    for u, v, d in G.edges(data=True):
        if d.get("relationship_type") == "spouse":
            if u in gen and v not in gen:
                gen[v] = gen[u]
            elif v in gen and u not in gen:
                gen[u] = gen[v]

    # Any remaining nodes in the graph that weren't reached: assign gen -1
    for n in G.nodes():
        if n not in gen:
            gen[n] = -1

    return gen


# ─── Graph filtering ─────────────────────────────────────────────────────────

def _filter_tree_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Keep only TREE_TYPES edges, remove isolates."""
    H = G.copy()
    to_remove = [(u, v) for u, v, d in H.edges(data=True)
                 if d.get("relationship_type", "") not in TREE_TYPES]
    H.remove_edges_from(to_remove)
    H.remove_nodes_from(list(nx.isolates(H)))
    return H


def _get_family_subgraph(G: nx.DiGraph, branch: str) -> nx.DiGraph:
    """Extract nodes of a branch + their direct family connections."""
    branch_nodes = {n for n, d in G.nodes(data=True) if d.get("family_branch") == branch}
    connected = set()
    for n in branch_nodes:
        for _, v, _ in G.edges(n, data=True):
            connected.add(v)
        for u, _, _ in G.in_edges(n, data=True):
            connected.add(u)
    return G.subgraph(branch_nodes | connected).copy()


# ─── AGraph rendering ────────────────────────────────────────────────────────

def _build_agraph(G: nx.DiGraph, gen: dict[str, int], title: str) -> pgv.AGraph:
    """Build a pygraphviz AGraph with rank constraints from a NetworkX DiGraph."""
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

    # Add nodes
    for n in G.nodes():
        data = G.nodes[n]
        branch = data.get("family_branch", "other")
        fillcolor = BRANCH_FILLCOLORS.get(branch, BRANCH_FILLCOLORS["other"])
        bordercolor = BRANCH_COLORS.get(branch, BRANCH_COLORS["other"])
        A.add_node(n, fillcolor=fillcolor, color=bordercolor, penwidth="2.0")

    # Add edges (deduplicate undirected)
    seen_undirected = set()
    for u, v, d in G.edges(data=True):
        rtype = d.get("relationship_type", "")
        is_directed = d.get("directed", False)

        if not is_directed:
            key = tuple(sorted([u, v]))
            if key in seen_undirected:
                continue
            seen_undirected.add(key)

        style_attrs = EDGE_STYLES.get(rtype, {})
        edge_attrs = dict(style_attrs)  # copy

        if not is_directed:
            edge_attrs["dir"] = "none"

        A.add_edge(u, v, **edge_attrs)

    # Create rank=same subgraphs for each generation
    gen_groups: dict[int, list[str]] = {}
    for n in G.nodes():
        g = gen.get(n, -1)
        if g >= 0:
            gen_groups.setdefault(g, []).append(n)

    for g in sorted(gen_groups.keys()):
        nodes = gen_groups[g]
        subg = A.add_subgraph(nodes, name=f"gen_{g}")
        subg.graph_attr["rank"] = "same"

    return A


def _render(A: pgv.AGraph, output_path: str) -> None:
    """Render AGraph to PNG."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.draw(str(out), prog="dot", format="png")
    print(f"  Saved: {output_path}")


# ─── Main pipeline ───────────────────────────────────────────────────────────

def visualize_family_trees(G: nx.DiGraph, output_dir: str) -> None:
    """Generate one family tree per branch + combined."""
    H = _filter_tree_edges(G)
    print(f"Tree subgraph: {len(H.nodes())} nodes, {len(H.edges())} edges")

    gen = _assign_generations(G)  # use full graph for generation assignment

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Combined tree
    A = _build_agraph(H, gen, "Arbre généalogique — La Fortune des Rougon")
    _render(A, str(out / "family_tree_all.png"))

    # Per-branch trees
    branches = sorted({d.get("family_branch", "other") for _, d in H.nodes(data=True)} - {"other"})
    for branch in branches:
        sub = _get_family_subgraph(H, branch)
        if len(sub.nodes()) > 1:
            A = _build_agraph(sub, gen, f"Arbre généalogique — Famille {branch}")
            _render(A, str(out / f"family_tree_{branch.lower()}.png"))


def main(input_path: str, output_dir: str, prog: str = "dot") -> None:
    G = nx.read_graphml(input_path)
    # GraphML reads everything as strings; convert boolean-like attributes
    for node in G.nodes():
        data = G.nodes[node]
        if isinstance(data.get("uncertain"), str):
            data["uncertain"] = data["uncertain"].lower() == "true"
    for u, v in G.edges():
        data = G.edges[u, v]
        if isinstance(data.get("directed"), str):
            data["directed"] = data["directed"].lower() == "true"

    visualize_family_trees(G, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate family tree visualizations")
    parser.add_argument("--input", default="output/graph.graphml", help="Input GraphML path")
    parser.add_argument("--output", default="output", help="Output directory for tree images")
    parser.add_argument("--prog", default="dot", help="Graphviz layout program (unused, kept for compat)")
    args = parser.parse_args()
    main(args.input, args.output, args.prog)
