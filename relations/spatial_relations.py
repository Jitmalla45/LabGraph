from utils.helpers import center, distance


def compute_relations(objects, max_relations=2):
    relations = []
    for obj_a in objects:
        neighbors = sorted(
            [obj for obj in objects if obj is not obj_a],
            key=lambda obj: distance(obj_a, obj),
        )[:max_relations]
        for obj_b in neighbors:
            c1 = center(obj_a["bbox"])
            c2 = center(obj_b["bbox"])
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            if abs(dx) > 30:
                predicate = "left_of" if dx < 0 else "right_of"
            elif abs(dy) > 30:
                predicate = "in_front_of" if dy < 0 else "behind"
            else:
                predicate = "near"
            relations.append((obj_a["id"], predicate, obj_b["id"]))
    return list(set(relations))
