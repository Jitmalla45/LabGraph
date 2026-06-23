from utils.helpers import compute_centroid, crop_with_padding, geometric_hash, letterbox_resize, new_uuid
from models.qwen_inference import predict_single


def parse_objects(annotation, image):
    objects = []
    width, height = image.size
    obj_id = 1
    for img_data in annotation.values():
        for region in img_data.get("regions", []):
            shape = region.get("shape_attributes", {})
            attrs = region.get("region_attributes", {})
            name = attrs.get("name", "object").lower()
            x = float(shape["x"])
            y = float(shape["y"])
            w = float(shape["width"])
            h = float(shape["height"])
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(width, int(x + w)), min(height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [x1, y1, x2, y2]
            area = (x2 - x1) * (y2 - y1)
            objects.append({
                "id": f"obj{obj_id}",
                "uuid": new_uuid(),
                "gt": name,
                "name": name,
                "bbox": bbox,
                "centroid": compute_centroid(bbox),
                "hash": geometric_hash(bbox),
                "is_landmark": area > 0.12 * (width * height),
                "attributes": {"is_on": False, "color": "xxx", "material": "xxx"},
            })
            obj_id += 1
    return objects


def identify_object(image, bbox, model, processor):
    crop = crop_with_padding(image, bbox, pad=10)
    crop = letterbox_resize(crop, (224, 224))
    return predict_single(model, processor, crop)


def relabel_objects(image, objects, model, processor):
    for obj in objects:
        pred, confidence = identify_object(image, obj["bbox"], model, processor)
        obj["pred"] = pred
        obj["confidence"] = confidence
        if pred in {"voltmeter", "ammeter", "power supply"}:
            obj["attributes"]["is_on"] = True
        print(f"GT: {obj['gt']} | Pred: {pred} | Conf: {confidence}")
    return objects


def relabel_with_cache(image, objects, known_objects, model, processor):
    known_by_hash = {obj["hash"]: obj for obj in known_objects}
    for obj in objects:
        known = known_by_hash.get(obj["hash"])
        if known:
            obj["pred"] = known.get("pred", known.get("name", "unknown"))
            obj["confidence"] = known.get("confidence", 0)
            obj["attributes"] = known.get("attributes", obj["attributes"]).copy()
            continue
        pred, confidence = identify_object(image, obj["bbox"], model, processor)
        obj["pred"] = pred
        obj["confidence"] = confidence
    return objects
