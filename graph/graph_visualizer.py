from pathlib import Path

import networkx as nx
import pydot
from PIL import Image, ImageDraw, ImageFont

from utils.helpers import graph_id, object_display_name, safe_label


def draw_labeled(image, objects, save_path=None):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{object_display_name(obj)} ({obj.get('confidence', 0)})"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 20)), label, fill="yellow", font=font)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    return img


def visualize_scene_graph(scene, save_path):
    graph = pydot.Dot(rankdir="LR")
    for obj in scene["objects"]:
        graph.add_node(pydot.Node(
            graph_id(obj["id"]),
            label=safe_label(object_display_name(obj)),
            shape="box",
            style="filled",
            fillcolor="#f4cccc",
        ))
    for index, rel in enumerate(scene["relations"]):
        rid = f"rel_{index}"
        graph.add_node(pydot.Node(
            rid,
            label=safe_label(rel["predicate"]),
            shape="ellipse",
            style="filled",
            fillcolor="#cfe2f3",
        ))
        graph.add_edge(pydot.Edge(graph_id(rel["subject"]), rid))
        obj = rel["object"]
        if isinstance(obj, str) and obj.startswith("obj"):
            graph.add_edge(pydot.Edge(rid, graph_id(obj)))
        else:
            val_id = graph_id(f"{obj}{index}")
            graph.add_node(pydot.Node(
                val_id,
                label=safe_label(obj),
                shape="ellipse",
                style="filled",
                fillcolor="#d9ead3",
            ))
            graph.add_edge(pydot.Edge(rid, val_id))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    graph.write_png(str(save_path))
    return Image.open(save_path)


def build_nx_graph(scene):
    graph = nx.Graph()
    for rel in scene["relations"]:
        if isinstance(rel["object"], str) and rel["object"].startswith("obj"):
            graph.add_edge(rel["subject"], rel["object"], relation=rel["predicate"])
    return graph
