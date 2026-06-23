import json
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
from PIL import Image

from change_detection.pixel_change_detection import create_difference_map, detect_changes
from config import PipelineConfig
from detection.visual_grounding import landmark_grounding
from detection.object_detection import parse_objects, relabel_objects, relabel_with_cache
from evaluation.qa_dataset_evaluation import evaluate_qa_dataset
from evaluation.save_outputs import save_pipeline_outputs
from evaluation.visualization import save_difference_map
from graph.graph_update import apply_incremental_update
from graph.graph_visualizer import build_nx_graph
from graph.scene_graph_builder import build_scene_graph
from models.model_loader import load_qwen_stack
from reasoning.k_hop_reasoning import k_hop_reasoning
from reasoning.landmark_selection import select_k_landmarks
from reasoning.query_engine import answer_landmark_based_query, answer_queries
from utils.helpers import (
    assign_unique_object_names,
    ensure_dir,
    object_display_name,
    object_matches_query,
    save_json,
)


# ---------------------------------------------------------------------
# User-configurable input section
# ---------------------------------------------------------------------
ORIGINAL_IMAGE = "Data/phy-5.jpg"
ANNOTATION_JSON = "Data/phy-5.json"
MODIFIED_IMAGE = "Data/opt5phy.png"
IMAGE_ID = "5"



QUERY_LIST = [
    "Where is the glasses with respect to battery?",
    "Where is the battery?",
]

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
LORA_PATH = "FINETUNED QWEN/chemistry_lora"
OUTPUT_DIR = "FINOUT"
QA_DATASET = "Data/QA_DATASET.json"


def load_inputs(original_image, modified_image, annotation_json):
    original = Image.open(original_image).convert("RGB")
    modified = Image.open(modified_image).convert("RGB")
    with open(annotation_json) as handle:
        annotation = json.load(handle)
    return original, modified, annotation


def _derive_image_id(original_image):
    return Path(original_image).stem


def summarize_changes(changes):
    return {
        "added": [object_display_name(obj) for obj in changes["added"]],
        "removed": [object_display_name(obj) for obj in changes["removed"]],
        "moved": [object_display_name(old) for old, _ in changes["moved"]],
        "changed": [object_display_name(old) for old, _ in changes["changed"]],
    }


def _object_label(obj):
    return object_display_name(obj)


def _find_objects_in_query(scene_graph, query):
    matches = []
    for obj in scene_graph["objects"]:
        if object_matches_query(obj, query):
            matches.append(obj)
    return matches


def _resolve_grounding_context(scene_graph, query):
    matches = _find_objects_in_query(scene_graph, query)
    if not matches:
        return None, None, None, "Target not found."
    target = matches[0]
    reference = next((obj for obj in matches[1:] if obj["id"] != target["id"]), None)
    graph = build_nx_graph(scene_graph)

    if reference is None:
        neighbors = list(graph.neighbors(target["id"]))
        if not neighbors:
            return target, None, None, "No reference found."
        reference_id = next(
            (
                node for node in neighbors
                if _object_label(next(obj for obj in scene_graph["objects"] if obj["id"] == node)) != _object_label(target)
            ),
            neighbors[0],
        )
        reference = next(obj for obj in scene_graph["objects"] if obj["id"] == reference_id)

    try:
        if graph.has_edge(target["id"], reference["id"]):
            relation = graph[target["id"]][reference["id"]]["relation"]
        else:
            path = nx.shortest_path(graph, target["id"], reference["id"])
            if len(path) < 2:
                return target, reference, None, "No relation found."
            relation = graph[target["id"]][path[1]]["relation"]
    except Exception:
        return target, reference, None, "No relation found between target and reference."
    return target, reference, relation, None


def _k_hop_reasoning_path(scene_graph, query, k=4):
    query_lower = query.lower()
    target = None
    for obj in scene_graph["objects"]:
        if object_matches_query(obj, query_lower):
            target = obj
            break
    if target is None:
        return []

    graph = build_nx_graph(scene_graph)
    current = target["id"]
    visited_nodes = set()
    visited_labels = {_object_label(target)}
    path = [{"object_id": target["id"], "object": _object_label(target)}]

    for _ in range(k):
        visited_nodes.add(current)
        candidate, best_score = None, -1
        for neighbor in graph.neighbors(current):
            if neighbor in visited_nodes:
                continue
            obj_n = next(obj for obj in scene_graph["objects"] if obj["id"] == neighbor)
            label = _object_label(obj_n)
            if label in visited_labels:
                continue
            x1, y1, x2, y2 = obj_n["bbox"]
            score = (3 if obj_n.get("is_landmark", False) else 0) + ((x2 - x1) * (y2 - y1)) / 10000
            if score > best_score:
                candidate, best_score = neighbor, score
        if candidate is None:
            break
        relation = graph[current][candidate]["relation"]
        obj_next = next(obj for obj in scene_graph["objects"] if obj["id"] == candidate)
        path.append({
            "relation": relation,
            "object_id": obj_next["id"],
            "object": _object_label(obj_next),
        })
        visited_labels.add(_object_label(obj_next))
        current = candidate
    return path


def _save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(str(text))


def _next_reasoning_dir(output_root, folder):
    base_dir = ensure_dir(Path(output_root) / folder)
    existing = [
        int(path.name.split("_")[-1])
        for path in base_dir.glob("query_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    ]
    next_index = max(existing, default=0) + 1
    return ensure_dir(base_dir / f"query_{next_index:03d}")


def run_manual_query_section(scene_graph, image, landmarks, output_root):
    print("\n================================================")
    print("REASONING MENU")
    print("================================================")
    print("Scene Graph Generated")
    print("Dataset Evaluation Complete")

    while True:
        print("\nChoose Reasoning Mode:")
        print("1. K-Hop Reasoning")
        print("2. Landmark-Based Reasoning")
        print("3. Visual Grounding")
        print("4. Exit")
        mode = input("Mode: ").strip()
        if mode == "4":
            print("Exiting reasoning menu.")
            return
        if mode not in {"1", "2", "3"}:
            print("Invalid mode. Please choose 1, 2, 3, or 4.")
            continue
        query = input("Enter Query: ").strip()
        if not query:
            print("No query entered.")
            continue

        if mode == "1":
            output_dir = _next_reasoning_dir(output_root, "k_hop")
            answer = k_hop_reasoning(scene_graph, query, k=4)
            result = {
                "mode": "k_hop",
                "query": query,
                "answer": answer,
                "reasoning_path": _k_hop_reasoning_path(scene_graph, query, k=4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            _save_text(output_dir / "query.txt", query)
            _save_text(output_dir / "answer.txt", answer)
            save_json(output_dir / "reasoning_path.json", result)
            save_json(output_dir / "result.json", result)
        elif mode == "2":
            output_dir = _next_reasoning_dir(output_root, "landmark")
            answer = answer_landmark_based_query(
                scene_graph,
                query,
                landmarks,
                image=image,
                output_path=output_dir / "landmark_visualization.png",
            )
            target = next((_object_label(obj) for obj in _find_objects_in_query(scene_graph, query)), None)
            result = {
                "mode": "k_landmark",
                "query": query,
                "answer": answer,
                "target": target,
                "landmarks": landmarks,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            _save_text(output_dir / "query.txt", query)
            _save_text(output_dir / "answer.txt", answer)
            save_json(output_dir / "landmark_result.json", result)
            save_json(output_dir / "result.json", result)
        else:
            output_dir = _next_reasoning_dir(output_root, "visual_grounding")
            target, reference, relation, error = _resolve_grounding_context(scene_graph, query)
            if error:
                answer = error
                result = {"mode": "visual_grounding", "query": query, "answer": answer}
            else:
                answer = landmark_grounding(image, target, reference, relation, save_path=output_dir / "grounded_image.png")
                result = {
                    "mode": "visual_grounding",
                    "query": query,
                    "answer": answer,
                    "target": target,
                    "reference": reference,
                    "relation": relation,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            _save_text(output_dir / "query.txt", query)
            _save_text(output_dir / "answer.txt", answer)
            save_json(output_dir / "grounding_result.json", result)
            save_json(output_dir / "result.json", result)

        print("\nReasoning Answer:")
        print(answer)
        print("Saved to:", output_dir)


def run_complete_pipeline(original_image, modified_image, query_list, annotation_json=ANNOTATION_JSON, image_id=IMAGE_ID):
    config = PipelineConfig(
        output_dir=Path(OUTPUT_DIR),
        base_model=BASE_MODEL,
        lora_path=LORA_PATH,
        image_id=image_id,
        query_list=query_list,
    )
    output_paths = config.output_paths
    output_dir = output_paths.prepare_standard_dirs()
    original, modified, annotation = load_inputs(original_image, modified_image, annotation_json)

    model, processor = load_qwen_stack(config)

    objects = parse_objects(annotation, original)
    objects = relabel_objects(original, objects, model, processor)
    objects = assign_unique_object_names(objects)
    scene_graph = build_scene_graph(
        original,
        objects,
        model=model,
        processor=processor,
        max_relations=config.max_spatial_relations,
        max_attributes=config.max_attributes,
    )

    new_objects = parse_objects(annotation, modified)
    new_objects = relabel_with_cache(modified, new_objects, objects, model, processor)
    new_objects = assign_unique_object_names(new_objects)
    changes = detect_changes(
        objects,
        new_objects,
        original,
        modified,
        iou_thresh=config.iou_threshold,
        pixel_thresh=config.pixel_threshold,
    )
    updated_scene_graph = apply_incremental_update(
        scene_graph,
        changes,
        max_relations=config.max_spatial_relations,
    )

    landmarks = select_k_landmarks(
        scene_graph,
        k=config.landmark_count,
        delta=config.landmark_delta,
    )
    query_results = answer_queries(scene_graph, query_list, original, landmarks, output_dir)

    diff = create_difference_map(original, modified)
    save_difference_map(diff, output_dir / "difference_map.png")

    evaluation = save_pipeline_outputs(config, original, scene_graph, updated_scene_graph)
    save_json(output_dir / "changes.json", summarize_changes(changes))
    save_json(output_dir / "landmarks.json", landmarks)
    save_json(output_dir / "query_results.json", query_results)
    dataset_image_id = _derive_image_id(original_image)
    qa_evaluation = evaluate_qa_dataset(
        dataset_path=QA_DATASET,
        image_id=dataset_image_id,
        original_scene_graph=scene_graph,
        updated_scene_graph=updated_scene_graph,
        image=original,
        changes=changes,
        output_root=output_dir,
        updated_image=modified,
    )

    print("\n================================================")
    print("ALL OUTPUTS SAVED SUCCESSFULLY")
    print("================================================")
    print("Folder:", output_dir)
    print("Object accuracy:", round(evaluation["metrics"]["object_accuracy"], 4))
    print("Micro F1:", round(evaluation["metrics"]["micro"]["f1_score"], 4))
    print("Macro F1:", round(evaluation["metrics"]["macro"]["f1_score"], 4))
    print("QA queries evaluated:", qa_evaluation.get("query_count", 0))
    print("================================================")
    return {
        "scene_graph": scene_graph,
        "updated_scene_graph": updated_scene_graph,
        "changes": changes,
        "landmarks": landmarks,
        "queries": query_results,
        "evaluation": evaluation,
        "qa_evaluation": qa_evaluation,
        "original_image": original,
        "output_root": output_dir,
    }


if __name__ == "__main__":
    results = run_complete_pipeline(
        original_image=ORIGINAL_IMAGE,
        modified_image=MODIFIED_IMAGE,
        query_list=QUERY_LIST,
        annotation_json=ANNOTATION_JSON,
        image_id=IMAGE_ID,
    )
    if sys.stdin.isatty():
        run_manual_query_section(
            results["scene_graph"],
            results["original_image"],
            results["landmarks"],
            results["output_root"],
        )
