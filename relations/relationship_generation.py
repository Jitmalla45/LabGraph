def tuple_relations_to_dicts(relations):
    return [
        {"subject": subject, "predicate": predicate, "object": obj}
        for subject, predicate, obj in relations
    ]
