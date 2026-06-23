import re

import networkx as nx
import numpy as np

from detection.visual_grounding import landmark_grounding
from graph.graph_visualizer import build_nx_graph
from reasoning.k_hop_reasoning import k_hop_reasoning
from utils.helpers import object_display_name, object_matches_query


def _find_objects_in_query(scene, query):
    matches = []
    for obj in scene["objects"]:
        if object_matches_query(obj, query):
            matches.append(obj)
    return matches


def answer_landmark_query(scene, query, image, output_dir=None):
    query = query.lower()
    matches = _find_objects_in_query(scene, query)
    if not matches:
        return "Target not found."
    target = matches[0]
    reference = next((obj for obj in matches[1:] if obj["id"] != target["id"]), None)
    graph = build_nx_graph(scene)

    if reference is None:
        neighbors = list(graph.neighbors(target["id"]))
        if not neighbors:
            return "No reference found."
        reference_id = next(
            (node for node in neighbors if object_display_name(next(obj for obj in scene["objects"] if obj["id"] == node)) != object_display_name(target)),
            neighbors[0],
        )
        reference = next(obj for obj in scene["objects"] if obj["id"] == reference_id)

    try:
        if graph.has_edge(target["id"], reference["id"]):
            relation = graph[target["id"]][reference["id"]]["relation"]
        else:
            path = nx.shortest_path(graph, target["id"], reference["id"])
            if len(path) < 2:
                return "No relation found."
            relation = graph[target["id"]][path[1]]["relation"]
    except Exception:
        return "No relation found between target and reference."

    save_path = output_dir / "visual_grounding.png" if output_dir else None
    return landmark_grounding(image, target, reference, relation, save_path=save_path)


def answer_landmark_based_query(scene, query, landmarks, image=None, output_path=None):
    query = query.lower()
    target = next(
        (
            obj for obj in scene["objects"]
            if object_matches_query(obj, query)
        ),
        None,
    )
    if target is None:
        return "Target not found."
    best_landmark = min(
        landmarks,
        key=lambda lm: np.linalg.norm(np.array(target["centroid"]) - np.array(lm["centroid"])),
    )
    dx = target["centroid"][0] - best_landmark["centroid"][0]
    dy = target["centroid"][1] - best_landmark["centroid"][1]
    relation = ("right of" if dx > 0 else "left of") if abs(dx) > abs(dy) else ("below" if dy > 0 else "above")
    text = f"{object_display_name(target)} is {relation} landmark {object_display_name(best_landmark)}"
    if image is not None:
        landmark_grounding(image, target, best_landmark, relation, save_path=output_path)
    return text


def answer_queries(scene, queries, image, landmarks, output_dir):
    results = {}
    for idx, query in enumerate(queries, start=1):
        results[query] = {
            "visual_grounding": answer_landmark_query(scene, query, image, output_dir if idx == 1 else None),
            "k_hop": k_hop_reasoning(scene, query, k=4),
            "landmark_based": answer_landmark_based_query(scene, query, landmarks, image=image, output_path=output_dir / f"landmark_query_{idx}.png"),
        }
    return results
