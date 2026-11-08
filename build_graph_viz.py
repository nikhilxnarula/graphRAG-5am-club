"""Build an interactive knowledge-graph page from the GraphRAG index output.

    python build_graph_viz.py                       # -> the-5am-club/graph.html
    python build_graph_viz.py --png graph.png       # also a static preview

Reads entities / relationships / communities / community_reports parquet files,
lays the graph out with networkx (clusters placed by community), and writes a
single self-contained HTML file (vis-network is inlined, falling back to a CDN tag).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

VIS_VERSION = "9.1.9"
VIS_URL = f"https://unpkg.com/vis-network@{VIS_VERSION}/standalone/umd/vis-network.min.js"
MAX_SLOTS = 8  # categorical palette has 8 hues; smaller communities fold into "Other"
DESC_CHARS = 420
EXPLODE = 3.2        # how far community centroids are pushed apart
CLUSTER_SHRINK = 1.6  # scale of each cluster's internal spread

# Reference categorical palette (light / dark), fixed order. Mirrors CSS in TEMPLATE.
PNG_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def pretty(name: str) -> str:
    out = " ".join(w.capitalize() for w in name.split(" "))
    return re.sub(r"(?<=\d )Am\b", "AM", out)


def load(root: Path):
    out = root / "output"
    ents = pd.read_parquet(out / "entities.parquet")
    rels = pd.read_parquet(out / "relationships.parquet")
    comms = pd.read_parquet(out / "communities.parquet")
    reps = pd.read_parquet(out / "community_reports.parquet")
    return ents, rels, comms, reps


def build(root: Path):
    ents, rels, comms, reps = load(root)

    # --- level-0 communities (top of the Leiden hierarchy) -> colour slots -------------
    l0 = comms[comms["level"] == 0].merge(
        reps[["community", "title", "summary"]], on="community", suffixes=("_raw", "")
    )
    l0 = l0.sort_values("size", ascending=False).reset_index(drop=True)
    id_to_title = dict(zip(ents["id"], ents["title"]))
    comm_of: dict[str, int] = {}
    for _, row in l0.iterrows():
        for eid in row["entity_ids"]:
            comm_of[id_to_title[eid]] = int(row["community"])
    slot_of_comm = {int(c): (i + 1 if i < MAX_SLOTS else 0) for i, c in enumerate(l0["community"])}

    # --- graph ----------------------------------------------------------------------------
    G = nx.Graph()
    for _, e in ents.iterrows():
        G.add_node(e["title"])
    for _, r in rels.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r["weight"]))

    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc).copy()

    # place community centres with a spring layout of the community meta-graph
    meta = nx.Graph()
    meta.add_nodes_from(int(c) for c in l0["community"])
    for u, v, d in H.edges(data=True):
        cu, cv = comm_of.get(u), comm_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            w = meta[cu][cv]["weight"] + math.log1p(d["weight"]) if meta.has_edge(cu, cv) else math.log1p(d["weight"])
            meta.add_edge(cu, cv, weight=w)
    centres = nx.spring_layout(meta, k=1.6, iterations=400, seed=11, weight="weight")

    import random

    rnd = random.Random(5)
    size_of = dict(zip(l0["community"].astype(int), l0["size"].astype(int)))
    init = {}
    for n in H.nodes:
        c = comm_of.get(n)
        cx, cy = centres[c] if c is not None else (0.0, 0.0)
        rad = 0.09 * math.sqrt(size_of.get(c, 10))
        ang = rnd.uniform(0, 2 * math.pi)
        init[n] = (cx + rad * math.cos(ang) * rnd.random(), cy + rad * math.sin(ang) * rnd.random())

    for u, v, d in H.edges(data=True):
        same = comm_of.get(u) == comm_of.get(v)
        d["w"] = math.log1p(d["weight"]) * (3.0 if same else 1.0)
    pos = nx.spring_layout(H, pos=init, k=0.045, iterations=120, seed=3, weight="w", scale=None)

    # "explode" the clusters: push each community's centroid outward so hubs of different
    # communities stop overlapping, keeping each cluster's internal shape.
    gx0 = sum(p[0] for p in pos.values()) / len(pos)
    gy0 = sum(p[1] for p in pos.values()) / len(pos)
    members: dict[int, list[str]] = {}
    for n in pos:
        members.setdefault(comm_of.get(n, -1), []).append(n)
    exploded = {}
    for c, ns in members.items():
        mx = sum(pos[n][0] for n in ns) / len(ns)
        my = sum(pos[n][1] for n in ns) / len(ns)
        for n in ns:
            exploded[n] = (
                gx0 + EXPLODE * (mx - gx0) + CLUSTER_SHRINK * (pos[n][0] - mx),
                gy0 + EXPLODE * (my - gy0) + CLUSTER_SHRINK * (pos[n][1] - my),
            )
    pos = exploded

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    k = 1900.0 / span
    cx0, cy0 = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    xy = {n: ((p[0] - cx0) * k, (p[1] - cy0) * k) for n, p in pos.items()}

    # fragments (everything outside the largest component): packed into a grid on the right
    rest = [c for c in nx.connected_components(G) if c is not lcc and not (c & lcc)]
    rest.sort(key=len, reverse=True)
    max_x = max(x for x, _ in xy.values())
    cell, cols = 110.0, 12
    for i, comp in enumerate(rest):
        gx = max_x + 250 + (i % cols) * cell
        gy = -600 + (i // cols) * cell
        if len(comp) == 1:
            xy[next(iter(comp))] = (gx, gy)
        else:
            sub = G.subgraph(comp)
            lp = nx.spring_layout(sub, seed=2, scale=50)
            for n, p in lp.items():
                xy[n] = (gx + float(p[0]), gy + float(p[1]))

    # --- payload ----------------------------------------------------------------------------
    index = {n: i for i, n in enumerate(G.nodes)}
    edesc = {}
    for _, r in rels.iterrows():
        edesc[(r["source"], r["target"])] = r["description"]
    nodes = []
    for _, e in ents.iterrows():
        t = e["title"]
        c = comm_of.get(t)
        nodes.append(
            {
                "id": index[t],
                "name": pretty(t),
                "type": e["type"],
                "deg": int(G.degree[t]),
                "freq": int(e["frequency"]),
                "comm": -1 if c is None else c,
                "slot": -1 if c is None else slot_of_comm[c],
                "desc": (e["description"] or "")[:DESC_CHARS],
                "x": round(xy[t][0], 1),
                "y": round(xy[t][1], 1),
            }
        )
    edges = [
        {
            "s": index[u],
            "t": index[v],
            "w": d["weight"],
            "desc": (edesc.get((u, v)) or edesc.get((v, u)) or "")[:DESC_CHARS],
        }
        for u, v, d in G.edges(data=True)
    ]
    communities = [
        {
            "id": int(r["community"]),
            "title": r["title"],
            "summary": r["summary"],
            "size": int(r["size"]),
            "slot": slot_of_comm[int(r["community"])],
        }
        for _, r in l0.iterrows()
    ]
    payload = {"nodes": nodes, "edges": edges, "communities": communities, "maxSlots": MAX_SLOTS}
    return payload, H, xy, comm_of, slot_of_comm


def static_png(path: Path, payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = {n["id"]: n for n in payload["nodes"]}
    fig, ax = plt.subplots(figsize=(16, 12), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for e in payload["edges"]:
        a, b = nodes[e["s"]], nodes[e["t"]]
        if a["comm"] < 0 or b["comm"] < 0:
            continue
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#898781", alpha=0.25, lw=0.4 + 0.25 * math.log1p(e["w"]), zorder=1)
    main = [n for n in nodes.values() if n["comm"] >= 0]
    ax.scatter(
        [n["x"] for n in main],
        [n["y"] for n in main],
        s=[18 + 22 * math.sqrt(n["deg"]) ** 1.5 for n in main],
        c=[PNG_PALETTE[n["slot"] - 1] if n["slot"] > 0 else "#898781" for n in main],
        edgecolors="#fcfcfb",
        linewidths=0.8,
        zorder=2,
    )
    for n in sorted(main, key=lambda n: -n["deg"])[:35]:
        ax.annotate(n["name"], (n["x"], n["y"]), fontsize=8, ha="center", va="bottom", color="#0b0b0b",
                    xytext=(0, 4 + math.sqrt(n["deg"])), textcoords="offset points")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())


def inline_vis() -> str:
    try:
        with urllib.request.urlopen(VIS_URL, timeout=30) as r:
            js = r.read().decode("utf-8")
        return f"<script>{js}</script>"
    except Exception as exc:  # offline: fall back to a CDN reference
        print(f"could not inline vis-network ({exc}); using CDN tag")
        return f'<script src="{VIS_URL}"></script>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="the-5am-club", type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--png", default=None, type=Path)
    args = ap.parse_args()

    payload, H, *_ = build(args.root)
    if args.png:
        static_png(args.png, payload)
        print("wrote", args.png)

    template = (Path(__file__).with_name("graph_template.html")).read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__VIS__*/", inline_vis()).replace("/*__DATA__*/", data)
    out = args.out or (args.root / "graph.html")
    out.write_text(html, encoding="utf-8")
    n_main = sum(1 for n in payload["nodes"] if n["comm"] >= 0)
    print(f"wrote {out}  ({len(payload['nodes'])} entities, {len(payload['edges'])} relationships, "
          f"{n_main} clustered, {len(payload['communities'])} communities)")


if __name__ == "__main__":
    main()
