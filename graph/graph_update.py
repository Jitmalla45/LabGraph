import copy

from relations.relationship_generation import tuple_relations_to_dicts
from relations.spatial_relations import compute_relations
from utils.helpers import assign_unique_object_names


def _object_edge(rel):
    return isinstance(rel.get("object"), str) and rel["object"].startswith("obj")


def _next_object_id(existing_ids):
    index = 1
    while f"obj{index}" in existing_ids:
        index += 1
    object_id = f"obj{index}"
    existing_ids.add(object_id)
    return object_id


def _updated_object(old_obj, new_obj):
    replacement = copy.deepcopy(new_obj)
    replacement["id"] = old_obj["id"]
    replacement["uuid"] = old_obj.get("uuid", replacement.get("uuid"))
    return replacement


def _dedupe_change_pairs(pairs, removed_ids):
    seen = set()
    deduped = []
    for old_obj, new_obj in pairs:
        old_id = old_obj.get("id")
        if old_id in removed_ids or old_id in seen:
            continue
        seen.add(old_id)
        deduped.append((old_obj, new_obj))
    return deduped


def apply_incremental_update(scene_graph, changes, max_relations=2):
    updated = copy.deepcopy(scene_graph)
    removed_ids = {obj["id"] for obj in changes.get("removed", [])}
    object_by_id = {
        obj["id"]: copy.deepcopy(obj)
        for obj in updated.get("objects", [])
        if obj["id"] not in removed_ids
    }

    for old_obj, new_obj in _dedupe_change_pairs(changes.get("moved", []), removed_ids):
        if old_obj["id"] in object_by_id:
            object_by_id[old_obj["id"]] = _updated_object(old_obj, new_obj)

    for old_obj, new_obj in _dedupe_change_pairs(changes.get("changed", []), removed_ids):
        if old_obj["id"] in object_by_id:
            object_by_id[old_obj["id"]] = _updated_object(old_obj, new_obj)

    existing_ids = set(object_by_id) | removed_ids
    added_objects = []
    for new_obj in changes.get("added", []):
        added = copy.deepcopy(new_obj)
        if added.get("id") in existing_ids:
            added["id"] = _next_object_id(existing_ids)
        else:
            existing_ids.add(added["id"])
        added_objects.append(added)

    updated_objects = list(object_by_id.values()) + added_objects
    assign_unique_object_names(updated_objects)

    preserved_attribute_relations = [
        copy.deepcopy(rel)
        for rel in scene_graph.get("relations", [])
        if not _object_edge(rel) and rel.get("subject") in object_by_id
    ]
    spatial_relations = tuple_relations_to_dicts(compute_relations(updated_objects, max_relations=max_relations))

    updated["objects"] = updated_objects
    updated["relations"] = spatial_relations + preserved_attribute_relations
    return updated
