from graph.graph_visualizer import build_nx_graph
from utils.helpers import object_display_name, object_matches_query


def k_hop_reasoning(scene, query, k=3):
    query = query.lower()
    target = None
    for obj in scene["objects"]:
        if object_matches_query(obj, query):
            target = obj
            break
    if target is None:
        return "Target not found."

    graph = build_nx_graph(scene)
    current = target["id"]
    visited_nodes = set()
    visited_labels = {object_display_name(target)}
    chain_text = object_display_name(target)
    first_relation = True

    for _ in range(k):
        visited_nodes.add(current)
        candidate, best_score = None, -1
        for neighbor in graph.neighbors(current):
            if neighbor in visited_nodes:
                continue
            obj_n = next(obj for obj in scene["objects"] if obj["id"] == neighbor)
            label = object_display_name(obj_n)
            if label in visited_labels:
                continue
            x1, y1, x2, y2 = obj_n["bbox"]
            score = (3 if obj_n.get("is_landmark", False) else 0) + ((x2 - x1) * (y2 - y1)) / 10000
            if score > best_score:
                candidate, best_score = neighbor, score
        if candidate is None:
            break
        relation = graph[current][candidate]["relation"].replace("_", " ")
        obj_next = next(obj for obj in scene["objects"] if obj["id"] == candidate)
        if first_relation:
            chain_text += f" is {relation} {object_display_name(obj_next)}"
            first_relation = False
        else:
            chain_text += f", which is {relation} {object_display_name(obj_next)}"
        visited_labels.add(object_display_name(obj_next))
        current = candidate
    return chain_text
