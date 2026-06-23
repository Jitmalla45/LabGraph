from attributes.attribute_extraction import generate_attribute_relations
from relations.relationship_generation import tuple_relations_to_dicts
from relations.spatial_relations import compute_relations


def build_scene_graph(image, objects, model=None, processor=None, max_relations=2, max_attributes=2):
    spatial = tuple_relations_to_dicts(compute_relations(objects, max_relations))
    attributes = generate_attribute_relations(
        image,
        objects,
        model=model,
        processor=processor,
        max_attributes=max_attributes,
    )
    return {"objects": objects, "relations": spatial + attributes}
