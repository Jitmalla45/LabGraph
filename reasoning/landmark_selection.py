import numpy as np


def select_k_landmarks(scene, k=3, delta=150):
    scored = []
    for obj in scene["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        area = (x2 - x1) * (y2 - y1)
        degree = sum(
            rel["subject"] == obj["id"] or rel["object"] == obj["id"]
            for rel in scene["relations"]
        )
        scored.append((0.7 * area + 0.3 * degree, obj))
    scored.sort(reverse=True, key=lambda item: item[0])
    landmarks = []
    for _, obj in scored:
        if all(np.linalg.norm(np.array(obj["centroid"]) - np.array(lm["centroid"])) >= delta for lm in landmarks):
            landmarks.append(obj)
        if len(landmarks) == k:
            break
    return landmarks
