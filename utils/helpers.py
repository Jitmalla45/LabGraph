import json
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_centroid(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def geometric_hash(bbox, precision=0):
    x1, y1, x2, y2 = bbox
    cx, cy = compute_centroid(bbox)
    return (
        round(cx, precision),
        round(cy, precision),
        round(x2 - x1, precision),
        round(y2 - y1, precision),
    )


def new_uuid():
    return str(uuid.uuid4())


def center(bbox):
    return compute_centroid(bbox)


def distance(obj_a, obj_b):
    return float(
        np.linalg.norm(
            np.array(center(obj_a["bbox"])) - np.array(center(obj_b["bbox"]))
        )
    )


def letterbox_resize(image, size=(224, 224), color=(128, 128, 128)):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", size, color)
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def crop_with_padding(image, bbox, pad=10):
    x1, y1, x2, y2 = bbox
    width, height = image.size
    return image.crop((
        max(0, int(x1 - pad)),
        max(0, int(y1 - pad)),
        min(width, int(x2 + pad)),
        min(height, int(y2 + pad)),
    ))


def normalize_label(label):
    label = str(label).lower().strip()
    label = re.sub(r"\d+", "", label)
    label = re.sub(r"[^a-zA-Z ]", " ", label)
    return re.sub(r"\s+", " ", label).strip()


def _normalize_object_name(label):
    label = str(label or "object").lower().strip()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return re.sub(r"_+", "_", label).strip("_") or "object"


def assign_unique_object_names(objects, source_key="pred", target_key="unique_name"):
    base_names = [
        _normalize_object_name(obj.get(source_key, obj.get("name", obj.get("gt", obj.get("id", "object")))))
        for obj in objects
    ]
    counts = {}
    for name in base_names:
        counts[name] = counts.get(name, 0) + 1

    seen = {}
    for obj, base_name in zip(objects, base_names):
        obj["base_name"] = base_name
        if counts[base_name] == 1:
            obj[target_key] = base_name
            continue
        seen[base_name] = seen.get(base_name, 0) + 1
        obj[target_key] = f"{base_name}_{seen[base_name]}"
    return objects


def object_display_name(obj):
    return obj.get("unique_name", obj.get("pred", obj.get("name", obj.get("gt", obj.get("id", "object")))))


def object_query_labels(obj):
    labels = [
        obj.get("unique_name", ""),
        obj.get("base_name", ""),
        obj.get("pred", ""),
        obj.get("name", ""),
        obj.get("gt", ""),
    ]
    return [str(label).lower().strip() for label in labels if label]


def object_matches_query(obj, query):
    query = str(query).lower()
    for label in object_query_labels(obj):
        if re.search(r"\b" + re.escape(label) + r"\b", query):
            return True
    return False


def graph_id(text):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(text))


def safe_label(text, max_len=15):
    return str(text).replace("\n", " ")[:max_len]


def save_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4, default=str)
