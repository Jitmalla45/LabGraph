import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import main as pipeline
from detection.visual_grounding import landmark_grounding
from reasoning.k_hop_reasoning import k_hop_reasoning
from reasoning.query_engine import answer_landmark_based_query
from utils.helpers import save_json


WEB_INPUT_DIR = Path("webapp") / "runs"
WEB_OUTPUT_DIR = Path("webapp") / "outputs"
LEGACY_ROOT_OUTPUT_DIR = Path("outputs")


def _safe_stem(name):
    stem = Path(name).stem or "upload"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)


def build_run_id(original_name):
    return _safe_stem(original_name)


def expected_output_root(run_id):
    return WEB_OUTPUT_DIR / run_id


def save_uploads(original_file, annotation_file, modified_file, run_id):
    run_dir = WEB_INPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "original": run_dir / original_file.name,
        "annotation": run_dir / annotation_file.name,
        "modified": run_dir / modified_file.name,
    }
    for uploaded, path in [
        (original_file, paths["original"]),
        (annotation_file, paths["annotation"]),
        (modified_file, paths["modified"]),
    ]:
        uploaded.seek(0)
        with open(path, "wb") as handle:
            shutil.copyfileobj(uploaded, handle)
    return paths


def _copy_missing_artifacts(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _sync_web_output_root(results, run_id):
    output_root = expected_output_root(run_id)
    returned_root = Path(results.get("output_root", output_root))
    if returned_root.exists() and returned_root.resolve() != output_root.resolve():
        _copy_missing_artifacts(returned_root, output_root)

    legacy_root = LEGACY_ROOT_OUTPUT_DIR / run_id
    if legacy_root.exists():
        _copy_missing_artifacts(legacy_root, output_root)

    results["output_root"] = output_root
    return results


@contextmanager
def _web_output_dir():
    previous = pipeline.OUTPUT_DIR
    pipeline.OUTPUT_DIR = str(WEB_OUTPUT_DIR)
    try:
        yield
    finally:
        pipeline.OUTPUT_DIR = previous


def run_pipeline_from_uploads(original_file, annotation_file, modified_file):
    run_id = build_run_id(original_file.name)
    paths = save_uploads(original_file, annotation_file, modified_file, run_id)
    with _web_output_dir():
        results = pipeline.run_complete_pipeline(
            original_image=str(paths["original"]),
            modified_image=str(paths["modified"]),
            annotation_json=str(paths["annotation"]),
            query_list=[],
            image_id=run_id,
        )
    return _sync_web_output_root(results, run_id)


def answer_session_query(results, strategy, query):
    scene_graph = results["scene_graph"]
    image = results["original_image"]
    landmarks = results["landmarks"]
    output_root = Path(results["output_root"])
    timestamp = datetime.now().isoformat(timespec="seconds")

    if strategy == "K-Hop Reasoning":
        output_dir = pipeline._next_reasoning_dir(output_root, "k_hop")
        answer = k_hop_reasoning(scene_graph, query, k=4)
        reasoning_path = pipeline._k_hop_reasoning_path(scene_graph, query, k=4)
        result = {
            "mode": "k_hop",
            "strategy": strategy,
            "query": query,
            "answer": answer,
            "reasoning_path": reasoning_path,
            "timestamp": timestamp,
        }
        visualization = None
        pipeline._save_text(output_dir / "query.txt", query)
        pipeline._save_text(output_dir / "answer.txt", answer)
        save_json(output_dir / "reasoning_path.json", result)
        save_json(output_dir / "result.json", result)
    elif strategy == "K-Landmark Reasoning":
        output_dir = pipeline._next_reasoning_dir(output_root, "landmark")
        visualization = output_dir / "landmark_visualization.png"
        answer = answer_landmark_based_query(
            scene_graph,
            query,
            landmarks,
            image=image,
            output_path=visualization,
        )
        target = next(
            (pipeline._object_label(obj) for obj in pipeline._find_objects_in_query(scene_graph, query)),
            None,
        )
        result = {
            "mode": "k_landmark",
            "strategy": strategy,
            "query": query,
            "answer": answer,
            "target": target,
            "landmarks": landmarks,
            "timestamp": timestamp,
        }
        pipeline._save_text(output_dir / "query.txt", query)
        pipeline._save_text(output_dir / "answer.txt", answer)
        save_json(output_dir / "landmark_result.json", result)
        save_json(output_dir / "result.json", result)
    else:
        output_dir = pipeline._next_reasoning_dir(output_root, "visual_grounding")
        visualization = output_dir / "grounded_image.png"
        target, reference, relation, error = pipeline._resolve_grounding_context(scene_graph, query)
        if error:
            answer = error
            visualization = None
            result = {
                "mode": "visual_grounding",
                "strategy": strategy,
                "query": query,
                "answer": answer,
                "timestamp": timestamp,
            }
        else:
            answer = landmark_grounding(image, target, reference, relation, save_path=visualization)
            result = {
                "mode": "visual_grounding",
                "strategy": strategy,
                "query": query,
                "answer": answer,
                "target": target,
                "reference": reference,
                "relation": relation,
                "timestamp": timestamp,
            }
        pipeline._save_text(output_dir / "query.txt", query)
        pipeline._save_text(output_dir / "answer.txt", answer)
        save_json(output_dir / "grounding_result.json", result)
        save_json(output_dir / "result.json", result)

    return {
        "answer": answer,
        "result": result,
        "output_dir": output_dir,
        "visualization": visualization if visualization and Path(visualization).exists() else None,
    }
