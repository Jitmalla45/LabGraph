import cv2
import numpy as np

from change_detection.object_correspondence import bbox_iou
from utils.helpers import normalize_label


def compare_regions(img_old, img_new, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop1 = np.array(img_old.crop((x1, y1, x2, y2)))
    crop2 = np.array(img_new.crop((x1, y1, x2, y2)))
    crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))
    return float(np.mean(np.abs(crop1.astype(float) - crop2.astype(float))))


def foreground_ratio(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = np.array(img.crop((x1, y1, x2, y2)))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    return float(np.sum(thresh > 0) / thresh.size)


def visual_content_score(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = np.array(img.crop((x1, y1, x2, y2)))
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    texture = np.std(gray) / 255.0
    saturation = np.std(crop, axis=2).mean() / 255.0
    return float((0.45 * edge_density) + (0.35 * texture) + (0.20 * saturation))


def create_difference_map(old_img, new_img):
    orig_np = np.array(old_img)
    new_np = np.array(new_img)
    if orig_np.shape != new_np.shape:
        new_np = cv2.resize(new_np, (orig_np.shape[1], orig_np.shape[0]))
    gray1 = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(new_np, cv2.COLOR_RGB2GRAY)
    return cv2.absdiff(gray1, gray2)


def _background_like(obj):
    labels = {
        normalize_label(obj.get("pred", "")),
        normalize_label(obj.get("name", "")),
        normalize_label(obj.get("gt", "")),
    }
    labels.discard("")
    return bool(labels & {"background", "blank", "empty", "none", "no object", "no_object"})


def _one_to_one_matches(old_objects, new_objects, iou_thresh):
    candidates = []
    for old_idx, old_obj in enumerate(old_objects):
        for new_idx, new_obj in enumerate(new_objects):
            iou = bbox_iou(old_obj["bbox"], new_obj["bbox"])
            if iou >= iou_thresh:
                candidates.append((iou, old_idx, new_idx))

    matches = {}
    matched_old = set()
    matched_new = set()
    for iou, old_idx, new_idx in sorted(candidates, reverse=True):
        if old_idx in matched_old or new_idx in matched_new:
            continue
        matches[old_idx] = (new_idx, iou)
        matched_old.add(old_idx)
        matched_new.add(new_idx)
    return matches, matched_old, matched_new


def detect_changes(old_objects, new_objects, old_img, new_img, iou_thresh=0.7, pixel_thresh=25):
    added, removed, moved, changed = [], [], [], []
    matches, matched_old, matched_new = _one_to_one_matches(old_objects, new_objects, iou_thresh)

    for old_idx, old_obj in enumerate(old_objects):
        if old_idx not in matched_old:
            removed.append(old_obj)
            continue
        new_idx, _ = matches[old_idx]
        new_obj = new_objects[new_idx]
        shift = np.linalg.norm(np.array(old_obj["centroid"]) - np.array(new_obj["centroid"]))
        old_score = visual_content_score(old_img, old_obj["bbox"])
        new_score = visual_content_score(new_img, new_obj["bbox"])
        old_occ = foreground_ratio(old_img, old_obj["bbox"])
        new_occ = foreground_ratio(new_img, new_obj["bbox"])

        removed_by_content = old_score > 0.035 and new_score < max(0.018, old_score * 0.35)
        removed_by_occupancy = old_occ > 0.4 and new_occ < 0.1
        if removed_by_content or removed_by_occupancy or _background_like(new_obj):
            removed.append(old_obj)
            continue

        if shift > 20:
            moved.append((old_obj, new_obj))
        if compare_regions(old_img, new_img, old_obj["bbox"]) > pixel_thresh:
            changed.append((old_obj, new_obj))

    for idx, new_obj in enumerate(new_objects):
        if idx not in matched_new:
            added.append(new_obj)
    return {"added": added, "removed": removed, "moved": moved, "changed": changed}
