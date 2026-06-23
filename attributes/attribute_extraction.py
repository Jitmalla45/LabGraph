import json
import re
from collections import Counter

import numpy as np

from models.qwen_inference import run_qwen_prompt
from utils.helpers import crop_with_padding, letterbox_resize


VALID_COLORS = {
    "black", "blue", "brown", "gray", "green", "orange", "pink", "purple",
    "red", "silver", "transparent", "white", "yellow",
}
VALID_MATERIALS = {
    "glass", "metal", "paper", "plastic", "rubber", "wood", "ceramic",
    "fabric", "liquid", "unknown",
}


def _dominant_color_name(crop):
    arr = np.asarray(crop.convert("RGB").resize((64, 64)))
    pixels = arr.reshape(-1, 3)
    brightness = pixels.mean(axis=1)
    pixels = pixels[(brightness > 25) & (brightness < 245)]
    if len(pixels) == 0:
        pixels = arr.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    palette = {
        "black": (20, 20, 20),
        "white": (235, 235, 235),
        "gray": (130, 130, 130),
        "red": (200, 40, 40),
        "green": (50, 160, 70),
        "blue": (50, 90, 190),
        "yellow": (220, 190, 50),
        "orange": (220, 120, 40),
        "brown": (130, 80, 40),
        "purple": (130, 70, 160),
        "pink": (220, 120, 170),
    }
    return min(palette, key=lambda name: np.linalg.norm(mean - np.array(palette[name])))


def _heuristic_material(object_name, crop):
    name = object_name.lower()
    if any(token in name for token in ["glass", "beaker", "tube", "lens"]):
        return "glass"
    if any(token in name for token in ["wire", "meter", "stand", "rod"]):
        return "metal"
    if any(token in name for token in ["paper", "book", "label"]):
        return "paper"
    if any(token in name for token in ["student", "cloth"]):
        return "fabric"
    arr = np.asarray(crop.convert("RGB"))
    if arr.std() < 18 and arr.mean() > 180:
        return "plastic"
    return "unknown"


def _parse_attribute_response(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "color": str(data.get("color", "")).lower().strip(),
                "material": str(data.get("material", "")).lower().strip(),
            }
        except json.JSONDecodeError:
            pass
    color = ""
    material = ""
    for token in re.split(r"[,;\n]", text.lower()):
        if "color" in token:
            color = token.split(":")[-1].strip()
        if "material" in token:
            material = token.split(":")[-1].strip()
    return {"color": color, "material": material}


def verify_attributes(attrs, fallback):
    color = attrs.get("color", "").split()[0] if attrs.get("color") else ""
    material = attrs.get("material", "").split()[0] if attrs.get("material") else ""
    return {
        "color": color if color in VALID_COLORS else fallback["color"],
        "material": material if material in VALID_MATERIALS else fallback["material"],
    }


def extract_object_attributes(image, obj, model=None, processor=None):
    crop = crop_with_padding(image, obj["bbox"], pad=4)
    model_crop = letterbox_resize(crop, (224, 224))
    fallback = {
        "color": _dominant_color_name(crop),
        "material": _heuristic_material(obj.get("pred", obj.get("name", "")), crop),
    }
    if model is None or processor is None:
        return fallback

    prompt = f"""
You are analyzing one cropped physics-lab object: {obj.get('pred', obj.get('name', 'object'))}.
Return only JSON with two keys: "color" and "material".
Use the visible object region only. Do not infer attributes from other objects.
If a property is unclear, use "unknown".
"""
    response = run_qwen_prompt(model, processor, model_crop, prompt, max_new_tokens=32)
    return verify_attributes(_parse_attribute_response(response), fallback)


def generate_attribute_relations(image, objects, model=None, processor=None, max_attributes=2):
    attr_rels = []
    seen_signatures = Counter()
    for obj in objects:
        if obj.get("pred") in {"object", "unknown"}:
            continue
        attrs = extract_object_attributes(image, obj, model, processor)
        obj["attributes"]["color"] = attrs["color"]
        obj["attributes"]["material"] = attrs["material"]
        signature = (attrs["color"], attrs["material"])
        seen_signatures[signature] += 1
        for key in ("color", "material")[:max_attributes]:
            attr_rels.append({
                "subject": obj["id"],
                "predicate": key,
                "object": obj["attributes"][key],
            })
    return attr_rels
