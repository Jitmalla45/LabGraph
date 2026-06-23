import csv
import copy
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt

from reasoning.k_hop_reasoning import k_hop_reasoning
from reasoning.landmark_selection import select_k_landmarks
from reasoning.query_engine import answer_landmark_based_query, answer_landmark_query
from utils.helpers import ensure_dir, object_display_name, save_json


QUERY_TYPES = ["spatial", "reference_based", "k_hop", "k_landmark", "change_detection"]
STRATA = [
    "simple_spatial",
    "single_hop_relational",
    "k_hop_relational",
    "k_landmark_reasoning",
    "change_detection",
]
COMPARABLE_STRATA = [
    "simple_spatial",
    "single_hop_relational",
    "k_hop_relational",
    "k_landmark_reasoning",
]

STRATUM_MODULES = {
    "simple_spatial": "scene_graph_spatial_query_engine",
    "single_hop_relational": "visual_grounding_reference_based_reasoning",
    "k_hop_relational": "k_hop_reasoning",
    "k_landmark_reasoning": "k_landmark_reasoning",
    "change_detection": "change_detection_graph_update_analysis",
}


def _normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_scores(ground_truth, prediction):
    gt_norm = _normalize_text(ground_truth)
    pred_norm = _normalize_text(prediction)
    gt_tokens = gt_norm.split()
    pred_tokens = pred_norm.split()
    exact_match = gt_norm == pred_norm
    if not gt_tokens or not pred_tokens:
        precision = 1.0 if not gt_tokens and not pred_tokens else 0.0
        recall = precision
        f1 = precision
    else:
        gt_counts = Counter(gt_tokens)
        pred_counts = Counter(pred_tokens)
        overlap = sum((gt_counts & pred_counts).values())
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "exact_match": exact_match,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "semantic_similarity": _semantic_similarity(gt_norm, pred_norm),
    }


def _semantic_similarity(ground_truth, prediction):
    gt_norm = _normalize_text(ground_truth)
    pred_norm = _normalize_text(prediction)
    if not gt_norm and not pred_norm:
        return 1.0
    if not gt_norm or not pred_norm:
        return 0.0
    return float(SequenceMatcher(None, gt_norm, pred_norm).ratio())


def _changes_to_answer(changes):
    parts = []
    added = [object_display_name(obj) for obj in changes.get("added", [])]
    removed = [object_display_name(obj) for obj in changes.get("removed", [])]
    moved = [object_display_name(old) for old, _ in changes.get("moved", [])]
    changed = [object_display_name(old) for old, _ in changes.get("changed", [])]
    if added:
        parts.append("Added: " + ", ".join(added))
    if removed:
        parts.append("Removed: " + ", ".join(removed))
    if moved:
        parts.append("Moved: " + ", ".join(moved))
    if changed:
        parts.append("Changed: " + ", ".join(changed))
    return "; ".join(parts) if parts else "No detected changes."


def _empty_changes():
    return {"added": [], "removed": [], "moved": [], "changed": []}


def _run_dataset_query(scene_graph, question, query_type, image, landmarks, changes=None):
    if query_type in {"spatial", "reference_based"}:
        return answer_landmark_query(scene_graph, question, image)
    if query_type == "k_hop":
        return k_hop_reasoning(scene_graph, question, k=4)
    if query_type == "k_landmark":
        return answer_landmark_based_query(scene_graph, question, landmarks, image=None)
    if query_type == "change_detection":
        return _changes_to_answer(changes or {})
    return k_hop_reasoning(scene_graph, question, k=4)


def _run_stratum_query(scene_graph, question, stratum, image, landmarks, changes=None):
    if stratum == "simple_spatial":
        return answer_landmark_query(scene_graph, question, image), STRATUM_MODULES[stratum]
    if stratum == "single_hop_relational":
        return answer_landmark_query(scene_graph, question, image), STRATUM_MODULES[stratum]
    if stratum == "k_hop_relational":
        return k_hop_reasoning(scene_graph, question, k=4), STRATUM_MODULES[stratum]
    if stratum == "k_landmark_reasoning":
        return answer_landmark_based_query(scene_graph, question, landmarks, image=None), STRATUM_MODULES[stratum]
    if stratum == "change_detection":
        return _changes_to_answer(changes or {}), STRATUM_MODULES[stratum]
    return k_hop_reasoning(scene_graph, question, k=4), "fallback_k_hop_reasoning"


def _aggregate_scores(score_records):
    if not score_records:
        return {
            "exact_match_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "semantic_similarity": 0.0,
            "count": 0,
        }
    return {
        "exact_match_accuracy": sum(1 for scores in score_records if scores["exact_match"]) / len(score_records),
        "precision": sum(scores["precision"] for scores in score_records) / len(score_records),
        "recall": sum(scores["recall"] for scores in score_records) / len(score_records),
        "f1": sum(scores["f1"] for scores in score_records) / len(score_records),
        "semantic_similarity": sum(scores.get("semantic_similarity", 0.0) for scores in score_records) / len(score_records),
        "count": len(score_records),
    }


def _aggregate_by_type(score_records):
    grouped = defaultdict(list)
    for scores in score_records:
        grouped[scores["query_type"]].append(scores)
    return {query_type: _aggregate_scores(grouped.get(query_type, [])) for query_type in QUERY_TYPES}


def _aggregate_by_stratum(score_records):
    grouped = defaultdict(list)
    for scores in score_records:
        grouped[scores["stratum"]].append(scores)
    return {stratum: _aggregate_scores(grouped.get(stratum, [])) for stratum in STRATA}


def _metric_triplet(metrics):
    return [metrics["precision"], metrics["recall"], metrics["f1"]]


def _metrics_identical(original_metrics, updated_metrics, tolerance=1e-12):
    return all(
        abs(original_metrics[name] - updated_metrics[name]) <= tolerance
        for name in ["precision", "recall", "f1"]
    )


def _graphs_have_same_content(original_scene_graph, updated_scene_graph):
    return json.dumps(original_scene_graph, sort_keys=True, default=str) == json.dumps(
        updated_scene_graph,
        sort_keys=True,
        default=str,
    )


def _build_diagnostics(
    original_predictions,
    updated_predictions,
    original_metrics,
    updated_metrics,
    original_scene_graph=None,
    updated_scene_graph=None,
    original_results=None,
    updated_results=None,
):
    prediction_pairs = list(zip(original_predictions, updated_predictions))
    differing_predictions = sum(1 for original, updated in prediction_pairs if original != updated)
    total_pairs = len(prediction_pairs)
    metrics_identical = bool(total_pairs and _metrics_identical(original_metrics, updated_metrics))
    shared_result_objects = any(
        original is updated
        for original, updated in zip(original_results or [], updated_results or [])
    )
    same_graph_object = (
        original_scene_graph is not None
        and updated_scene_graph is not None
        and original_scene_graph is updated_scene_graph
    )
    same_object_list = (
        original_scene_graph is not None
        and updated_scene_graph is not None
        and original_scene_graph.get("objects") is updated_scene_graph.get("objects")
    )
    same_relation_list = (
        original_scene_graph is not None
        and updated_scene_graph is not None
        and original_scene_graph.get("relations") is updated_scene_graph.get("relations")
    )
    same_graph_content = (
        original_scene_graph is not None
        and updated_scene_graph is not None
        and _graphs_have_same_content(original_scene_graph, updated_scene_graph)
    )
    warnings = []
    if total_pairs and differing_predictions == 0:
        warnings.append(
            "WARNING:\n"
            "Original and Updated graph predictions are identical for all queries. "
            "Evaluation comparison may not be meaningful."
        )
    if metrics_identical:
        warnings.append(
            "WARNING:\n"
            "Evaluation metrics for original and updated graphs are identical. "
            "Please verify graph update pipeline."
        )
    if shared_result_objects:
        warnings.append(
            "WARNING: Original and updated result entries share object references. "
            "Evaluation result storage may not be independent."
        )
    if same_graph_object:
        warnings.append(
            "WARNING: Original and updated scene graph inputs reference the same object in memory."
        )
    elif same_object_list or same_relation_list:
        warnings.append(
            "WARNING: Original and updated scene graph inputs share mutable object/relation lists."
        )
    if same_graph_content:
        warnings.append(
            "WARNING: Original and updated scene graph content is identical before evaluation."
        )
    return {
        "number_of_original_predictions": len(original_predictions),
        "number_of_updated_predictions": len(updated_predictions),
        "number_of_differing_predictions": differing_predictions,
        "percentage_of_changed_answers": float((differing_predictions / total_pairs) * 100) if total_pairs else 0.0,
        "all_predictions_identical": bool(total_pairs and differing_predictions == 0),
        "all_metrics_identical": metrics_identical,
        "original_updated_result_objects_share_memory": shared_result_objects,
        "original_updated_graphs_same_object": same_graph_object,
        "original_updated_object_lists_same_object": same_object_list,
        "original_updated_relation_lists_same_object": same_relation_list,
        "original_updated_graph_content_identical": same_graph_content,
        "warnings": warnings,
    }


def _print_diagnostics(diagnostics):
    print("\n================================================")
    print("QA EVALUATION DIAGNOSTICS")
    print("================================================")
    print("Number of original predictions:", diagnostics["number_of_original_predictions"])
    print("Number of updated predictions:", diagnostics["number_of_updated_predictions"])
    print("Number of differing predictions:", diagnostics["number_of_differing_predictions"])
    print("Percentage of changed answers:", round(diagnostics["percentage_of_changed_answers"], 2))
    for warning in diagnostics["warnings"]:
        print(warning)
    print("================================================")


def _plot_original_vs_updated(summary, path, title):
    ensure_dir(Path(path).parent)
    names = ["Precision", "Recall", "F1"]
    original = _metric_triplet(summary["original"])
    updated = _metric_triplet(summary["updated"])
    x = range(len(names))
    width = 0.35
    plt.figure(figsize=(9, 6))
    plt.bar([i - width / 2 for i in x], original, width, label="Original")
    plt.bar([i + width / 2 for i in x], updated, width, label="Updated")
    plt.xticks(list(x), names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_precision_recall_f1(summary, path):
    ensure_dir(Path(path).parent)
    names = ["Precision", "Recall", "F1"]
    updated = _metric_triplet(summary["updated"])
    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, updated, color=["#3b82f6", "#10b981", "#f59e0b"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Updated Graph QA Metrics")
    for bar, value in zip(bars, updated):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.03, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_query_type_breakdown(summary, path):
    ensure_dir(Path(path).parent)
    labels = QUERY_TYPES
    original = [summary["by_query_type"]["original"][label]["f1"] for label in labels]
    updated = [summary["by_query_type"]["updated"][label]["f1"] for label in labels]
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], original, width, label="Original F1")
    plt.bar([i + width / 2 for i in x], updated, width, label="Updated F1")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Query Type Breakdown")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_evaluation_summary(summary, path):
    ensure_dir(Path(path).parent)
    names = ["Precision", "Recall", "F1"]
    original = _metric_triplet(summary["original"])
    updated = _metric_triplet(summary["updated"])
    diagnostics = summary.get("diagnostics", {})
    changed_pct = diagnostics.get("percentage_of_changed_answers", 0.0)
    same_pct = max(0.0, 100.0 - changed_pct)
    x = range(len(names))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].bar([i - width / 2 for i in x], original, width, label="Original")
    axes[0].bar([i + width / 2 for i in x], updated, width, label="Updated")
    axes[0].set_xticks(list(x), names)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("QA Metrics")
    axes[0].legend()

    bars = axes[1].bar(["Same", "Changed"], [same_pct, changed_pct], color=["#64748b", "#14b8a6"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Answers (%)")
    axes[1].set_title("Prediction Change")
    for bar, value in zip(bars, [same_pct, changed_pct]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 3, 98),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    fig.suptitle("QA Evaluation Summary")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_stratum_performance(stratum_metrics, path):
    ensure_dir(Path(path).parent)
    labels = STRATA
    precision = [stratum_metrics[label]["precision"] for label in labels]
    recall = [stratum_metrics[label]["recall"] for label in labels]
    f1 = [stratum_metrics[label]["f1"] for label in labels]
    x = range(len(labels))
    width = 0.25
    plt.figure(figsize=(13, 6))
    plt.bar([i - width for i in x], precision, width, label="Precision")
    plt.bar(list(x), recall, width, label="Recall")
    plt.bar([i + width for i in x], f1, width, label="F1")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Stratum Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_stratum_original_vs_updated(original_metrics, updated_metrics, path):
    ensure_dir(Path(path).parent)
    labels = COMPARABLE_STRATA
    original = [original_metrics[label]["f1"] for label in labels]
    updated = [updated_metrics[label]["f1"] for label in labels]
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], original, width, label="Original Graph F1")
    plt.bar([i + width / 2 for i in x], updated, width, label="Updated Graph F1")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Original vs Updated Graph by Stratum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_stratum_distribution(stratum_counts, path):
    ensure_dir(Path(path).parent)
    labels = STRATA
    counts = [stratum_counts.get(label, 0) for label in labels]
    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, counts, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Question Count")
    plt.title("Reasoning Category Distribution")
    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            str(value),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _write_csv(path, rows):
    ensure_dir(Path(path).parent)
    if not rows:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_legacy_qa_dataset(
    dataset_path,
    image_id,
    original_scene_graph,
    updated_scene_graph,
    image,
    changes,
    output_root,
    updated_image=None,
):
    dataset_path = Path(dataset_path)
    output_dir = ensure_dir(Path(output_root) / "dataset_evaluation")
    if not dataset_path.exists():
        save_json(output_dir / "evaluation_summary.json", {"image_id": image_id, "query_count": 0})
        return {"image_id": image_id, "query_count": 0, "output_dir": str(output_dir)}

    with open(dataset_path) as handle:
        dataset = json.load(handle)

    original_image = image
    updated_image = updated_image if updated_image is not None else image
    entries = [entry for entry in dataset if entry.get("image_id") == image_id]
    original_eval_graph = copy.deepcopy(original_scene_graph)
    updated_eval_graph = copy.deepcopy(updated_scene_graph)
    original_landmarks = select_k_landmarks(original_eval_graph)
    updated_landmarks = select_k_landmarks(updated_eval_graph)
    rows = []
    original_results = []
    updated_results = []
    original_predictions = []
    updated_predictions = []
    original_score_records = []
    updated_score_records = []

    for entry in entries:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        query_type = entry.get("type", "unknown")
        original_changes = _empty_changes() if query_type == "change_detection" else changes
        updated_changes = changes
        original_answer = _run_dataset_query(
            original_eval_graph,
            question,
            query_type,
            original_image,
            original_landmarks,
            changes=original_changes,
        )
        updated_answer = _run_dataset_query(
            updated_eval_graph,
            question,
            query_type,
            updated_image,
            updated_landmarks,
            changes=updated_changes,
        )
        original_predictions.append(original_answer)
        updated_predictions.append(updated_answer)
        original_scores = _token_scores(answer, original_answer)
        updated_scores = _token_scores(answer, updated_answer)
        original_score_records.append({**original_scores, "query_type": query_type})
        updated_score_records.append({**updated_scores, "query_type": query_type})
        row = {
            "question": question,
            "ground_truth_answer": answer,
            "predicted_original_answer": original_answer,
            "predicted_updated_answer": updated_answer,
            "query_type": query_type,
            "original_exact_match": original_scores["exact_match"],
            "original_precision": original_scores["precision"],
            "original_recall": original_scores["recall"],
            "original_f1": original_scores["f1"],
            "updated_exact_match": updated_scores["exact_match"],
            "updated_precision": updated_scores["precision"],
            "updated_recall": updated_scores["recall"],
            "updated_f1": updated_scores["f1"],
            "exact_match": updated_scores["exact_match"],
            "precision": updated_scores["precision"],
            "recall": updated_scores["recall"],
            "f1": updated_scores["f1"],
        }
        rows.append(row)
        original_results.append({**entry, "predicted_answer_original": original_answer, **original_scores})
        updated_results.append({**entry, "predicted_answer_updated": updated_answer, **updated_scores})

    original_metrics = _aggregate_scores(original_score_records)
    updated_metrics = _aggregate_scores(updated_score_records)
    diagnostics = _build_diagnostics(
        original_predictions,
        updated_predictions,
        original_metrics,
        updated_metrics,
        original_scene_graph=original_eval_graph,
        updated_scene_graph=updated_eval_graph,
        original_results=original_results,
        updated_results=updated_results,
    )
    summary = {
        "image_id": image_id,
        "query_count": len(rows),
        "original": original_metrics,
        "updated": updated_metrics,
        "by_query_type": {
            "original": _aggregate_by_type(original_score_records),
            "updated": _aggregate_by_type(updated_score_records),
        },
        "diagnostics": diagnostics,
    }

    _print_diagnostics(diagnostics)
    _write_csv(output_dir / "qa_results.csv", rows)
    save_json(output_dir / "qa_results.json", rows)
    save_json(output_dir / "original_graph_results.json", original_results)
    save_json(output_dir / "updated_graph_results.json", updated_results)
    save_json(output_dir / "evaluation_summary.json", summary)
    save_json(output_dir / "evaluation_diagnostics.json", diagnostics)
    _plot_precision_recall_f1(summary, output_dir / "precision_recall_f1.png")
    _plot_query_type_breakdown(summary, output_dir / "query_type_breakdown.png")
    _plot_original_vs_updated(summary, output_dir / "original_vs_updated.png", "Original vs Updated")
    _plot_evaluation_summary(summary, output_dir / "evaluation_summary.png")
    return {**summary, "output_dir": str(output_dir)}


def _new_query_row(
    entry,
    stratum,
    predicted_answer,
    scores,
    selected_module,
    original_answer=None,
    updated_answer=None,
    original_scores=None,
    updated_scores=None,
    original_module=None,
    updated_module=None,
    graph_scope="updated",
):
    return {
        "question": entry.get("question", ""),
        "ground_truth": entry.get("answer", ""),
        "predicted_answer": predicted_answer,
        "predicted_original_answer": original_answer,
        "predicted_updated_answer": updated_answer,
        "stratum": stratum,
        "generation_method": entry.get("generation_method", ""),
        "ground_truth_source": entry.get("ground_truth_source", ""),
        "selected_reasoning_module": selected_module,
        "original_reasoning_module": original_module,
        "updated_reasoning_module": updated_module,
        "graph_scope": graph_scope,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "semantic_similarity": scores["semantic_similarity"],
        "evaluation_scores": {
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "semantic_similarity": scores["semantic_similarity"],
        },
        "original_evaluation_scores": original_scores,
        "updated_evaluation_scores": updated_scores,
    }


def _strip_internal_metric_fields(metrics):
    return {
        stratum: {
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "semantic_similarity": values["semantic_similarity"],
            "count": values.get("count", 0),
        }
        for stratum, values in metrics.items()
    }


def _remove_legacy_evaluation_plots(output_dir):
    # Preserve existing artifacts for backward compatibility.
    return


def _evaluate_stratum_qa_dataset(
    dataset_path,
    image_id,
    original_scene_graph,
    updated_scene_graph,
    image,
    changes,
    output_root,
    updated_image=None,
):
    output_dir = ensure_dir(Path(output_root) / "dataset_evaluation")
    with open(dataset_path) as handle:
        dataset = json.load(handle)

    original_image = image
    updated_image = updated_image if updated_image is not None else image
    entries = [entry for entry in dataset if entry.get("image_id") == image_id]
    original_eval_graph = copy.deepcopy(original_scene_graph)
    updated_eval_graph = copy.deepcopy(updated_scene_graph)
    original_landmarks = select_k_landmarks(original_eval_graph)
    updated_landmarks = select_k_landmarks(updated_eval_graph)

    query_results = []
    execution_log = []
    primary_score_records = []
    original_score_records = []
    updated_score_records = []
    original_predictions = []
    updated_predictions = []
    original_result_refs = []
    updated_result_refs = []

    for entry in entries:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        stratum = entry.get("stratum", "unknown")
        if stratum == "change_detection":
            updated_answer, updated_module = _run_stratum_query(
                updated_eval_graph,
                question,
                stratum,
                updated_image,
                updated_landmarks,
                changes=changes,
            )
            updated_scores = _token_scores(answer, updated_answer)
            primary_scores = updated_scores
            primary_record = {**primary_scores, "stratum": stratum}
            primary_score_records.append(primary_record)
            updated_score_records.append({**updated_scores, "stratum": stratum})
            row = _new_query_row(
                entry,
                stratum,
                updated_answer,
                primary_scores,
                updated_module,
                updated_answer=updated_answer,
                updated_scores=updated_scores,
                updated_module=updated_module,
                graph_scope="updated_only",
            )
        else:
            original_answer, original_module = _run_stratum_query(
                original_eval_graph,
                question,
                stratum,
                original_image,
                original_landmarks,
                changes=_empty_changes(),
            )
            updated_answer, updated_module = _run_stratum_query(
                updated_eval_graph,
                question,
                stratum,
                updated_image,
                updated_landmarks,
                changes=changes,
            )
            original_scores = _token_scores(answer, original_answer)
            updated_scores = _token_scores(answer, updated_answer)
            original_predictions.append(original_answer)
            updated_predictions.append(updated_answer)
            primary_scores = updated_scores
            primary_score_records.append({**primary_scores, "stratum": stratum})
            original_score_records.append({**original_scores, "stratum": stratum})
            updated_score_records.append({**updated_scores, "stratum": stratum})
            original_result_refs.append({**entry, "predicted_answer": original_answer, **original_scores})
            updated_result_refs.append({**entry, "predicted_answer": updated_answer, **updated_scores})
            row = _new_query_row(
                entry,
                stratum,
                updated_answer,
                primary_scores,
                updated_module,
                original_answer=original_answer,
                updated_answer=updated_answer,
                original_scores=original_scores,
                updated_scores=updated_scores,
                original_module=original_module,
                updated_module=updated_module,
                graph_scope="original_and_updated",
            )

        query_results.append(row)
        execution_log.append({
            "question": row["question"],
            "stratum": row["stratum"],
            "selected_reasoning_module": row["selected_reasoning_module"],
            "ground_truth": row["ground_truth"],
            "predicted_answer": row["predicted_answer"],
            "evaluation_scores": row["evaluation_scores"],
            "graph_scope": row["graph_scope"],
        })

    primary_by_stratum = _aggregate_by_stratum(primary_score_records)
    original_by_stratum = _aggregate_by_stratum(original_score_records)
    updated_by_stratum = _aggregate_by_stratum(updated_score_records)
    original_metrics = _aggregate_scores(original_score_records)
    updated_metrics = _aggregate_scores([
        record for record in updated_score_records
        if record["stratum"] in COMPARABLE_STRATA
    ])
    diagnostics = _build_diagnostics(
        original_predictions,
        updated_predictions,
        original_metrics,
        updated_metrics,
        original_scene_graph=original_eval_graph,
        updated_scene_graph=updated_eval_graph,
        original_results=original_result_refs,
        updated_results=updated_result_refs,
    )
    stratum_counts = dict(Counter(entry.get("stratum", "unknown") for entry in entries))
    stratum_metrics = _strip_internal_metric_fields(primary_by_stratum)
    original_vs_updated_metrics = {
        "original": _strip_internal_metric_fields(original_by_stratum),
        "updated": _strip_internal_metric_fields(updated_by_stratum),
    }
    summary = {
        "image_id": image_id,
        "dataset_format": "stratum",
        "query_count": len(query_results),
        "stratum_counts": stratum_counts,
        "stratum_metrics": stratum_metrics,
        "original_vs_updated_metrics": original_vs_updated_metrics,
        "diagnostics": diagnostics,
    }

    _print_diagnostics(diagnostics)
    _remove_legacy_evaluation_plots(output_dir)
    _write_csv(output_dir / "query_results.csv", query_results)
    save_json(output_dir / "query_results.json", query_results)
    save_json(output_dir / "query_execution_log.json", execution_log)
    save_json(output_dir / "stratum_metrics.json", stratum_metrics)
    save_json(output_dir / "original_vs_updated_metrics.json", original_vs_updated_metrics)
    save_json(output_dir / "evaluation_summary.json", summary)
    save_json(output_dir / "evaluation_diagnostics.json", diagnostics)
    _plot_stratum_performance(primary_by_stratum, output_dir / "stratum_performance.png")
    _plot_stratum_performance(primary_by_stratum, output_dir / "precision_recall_f1.png")
    _plot_stratum_original_vs_updated(
        original_by_stratum,
        updated_by_stratum,
        output_dir / "original_vs_updated_by_stratum.png",
    )
    _plot_stratum_distribution(stratum_counts, output_dir / "query_distribution.png")
    return {**summary, "output_dir": str(output_dir)}


def evaluate_qa_dataset(
    dataset_path,
    image_id,
    original_scene_graph,
    updated_scene_graph,
    image,
    changes,
    output_root,
    updated_image=None,
):
    dataset_path = Path(dataset_path)
    output_dir = ensure_dir(Path(output_root) / "dataset_evaluation")
    if not dataset_path.exists():
        save_json(output_dir / "evaluation_summary.json", {"image_id": image_id, "query_count": 0})
        return {"image_id": image_id, "query_count": 0, "output_dir": str(output_dir)}

    with open(dataset_path) as handle:
        dataset = json.load(handle)

    if any("stratum" in entry for entry in dataset):
        return _evaluate_stratum_qa_dataset(
            dataset_path=dataset_path,
            image_id=image_id,
            original_scene_graph=original_scene_graph,
            updated_scene_graph=updated_scene_graph,
            image=image,
            changes=changes,
            output_root=output_root,
            updated_image=updated_image,
        )
    return _evaluate_legacy_qa_dataset(
        dataset_path=dataset_path,
        image_id=image_id,
        original_scene_graph=original_scene_graph,
        updated_scene_graph=updated_scene_graph,
        image=image,
        changes=changes,
        output_root=output_root,
        updated_image=updated_image,
    )
