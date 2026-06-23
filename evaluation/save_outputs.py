import pandas as pd

from evaluation.metrics import evaluate_objects
from evaluation.visualization import save_confusion_matrix, save_metrics_plot
from graph.graph_visualizer import draw_labeled, visualize_scene_graph
from utils.helpers import ensure_dir, object_display_name, save_json


def serialize_scene_graph(scene_graph, graph_name, image_id):
    nodes = []
    for obj in scene_graph.get("objects", []):
        nodes.append({
            "id": obj.get("id"),
            "label": object_display_name(obj),
            "prediction": obj.get("pred", obj.get("name", obj.get("gt", "unknown"))),
            "unique_name": obj.get("unique_name"),
            "base_name": obj.get("base_name"),
            "ground_truth": obj.get("gt"),
            "name": obj.get("name"),
            "confidence": obj.get("confidence"),
            "bbox": obj.get("bbox"),
            "centroid": obj.get("centroid"),
            "attributes": obj.get("attributes", []),
            "is_landmark": obj.get("is_landmark", False),
        })

    edges = []
    attributes = []
    relationships = []
    for rel in scene_graph.get("relations", []):
        record = {
            "subject": rel.get("subject"),
            "predicate": rel.get("predicate"),
            "object": rel.get("object"),
        }
        relationships.append(record)
        if isinstance(rel.get("object"), str) and rel["object"].startswith("obj"):
            edges.append(record)
        else:
            attributes.append(record)

    return {
        "metadata": {
            "graph_name": graph_name,
            "image_id": image_id,
            "object_count": len(nodes),
            "relationship_count": len(relationships),
            "edge_count": len(edges),
            "attribute_count": len(attributes),
        },
        "nodes": nodes,
        "attributes": attributes,
        "edges": edges,
        "relationships": relationships,
    }


def save_scene_graph_exports(config, scene_graph, updated_scene_graph):
    for graph_dir in [
        config.output_paths.subdir("scene_graphs"),
        config.output_paths.subdir("scene_graph"),
    ]:
        original_png = graph_dir / "original_scene_graph.png"
        updated_png = graph_dir / "updated_scene_graph.png"
        visualize_scene_graph(scene_graph, original_png)
        visualize_scene_graph(updated_scene_graph, updated_png)
        save_json(
            graph_dir / "original_scene_graph.json",
            serialize_scene_graph(scene_graph, "original", config.image_id),
        )
        save_json(
            graph_dir / "updated_scene_graph.json",
            serialize_scene_graph(updated_scene_graph, "updated", config.image_id),
        )


def save_pipeline_outputs(config, image, scene_graph, updated_scene_graph):
    output_dir = config.output_paths.root
    evaluation = evaluate_objects(scene_graph["objects"])
    metrics = evaluation["metrics"]

    rows = [{
        "id": obj["id"],
        "ground_truth": obj["gt"],
        "prediction": obj.get("pred", "unknown"),
        "unique_name": obj.get("unique_name", obj.get("pred", "unknown")),
        "base_name": obj.get("base_name", obj.get("pred", "unknown")),
        "confidence": obj.get("confidence", 0),
        "bbox": obj["bbox"],
        "centroid": obj["centroid"],
        "attributes": obj["attributes"],
    } for obj in scene_graph["objects"]]

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "full_evaluation.json", {"metrics": metrics, "objects": rows})
    save_json(output_dir / "classification_report.json", evaluation["classification_report"])
    ensure_dir(output_dir)
    pd.DataFrame(rows).to_csv(output_dir / "evaluation_results.csv", index=False)
    draw_labeled(image, scene_graph["objects"], save_path=output_dir / "labeled_image.png")
    visualize_scene_graph(scene_graph, output_dir / f"{config.image_id}_scene_graph_original.png")
    visualize_scene_graph(updated_scene_graph, output_dir / f"{config.image_id}_scene_graph_optimized.png")
    save_json(output_dir / "scene_graph.json", scene_graph)
    save_json(output_dir / "scene_graph_optimized.json", updated_scene_graph)
    save_scene_graph_exports(config, scene_graph, updated_scene_graph)
    save_confusion_matrix(
        evaluation["confusion_matrix"],
        evaluation["labels"],
        output_dir / "confusion_matrix.png",
    )
    save_metrics_plot(metrics, output_dir / "metrics_visualization.png")
    return evaluation
