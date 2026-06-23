from pathlib import Path

from utils.helpers import ensure_dir


class OutputPathManager:
    def __init__(self, base_dir, image_id):
        self.base_dir = Path(base_dir)
        self.image_id = str(image_id)
        self.image_dir = self.base_dir / self.image_id

    @property
    def root(self):
        return ensure_dir(self.image_dir)

    def subdir(self, name):
        return ensure_dir(self.root / name)

    def file(self, *parts):
        path = self.root.joinpath(*parts)
        ensure_dir(path.parent)
        return path

    def subfile(self, subdir, filename):
        return self.file(subdir, filename)

    def prepare_standard_dirs(self):
        for name in [
            "scene_graph",
            "scene_graphs",
            "visual_grounding",
            "landmark",
            "landmark_reasoning",
            "k_hop",
            "khop_reasoning",
            "dataset_evaluation",
            "change_detection",
            "visualizations",
            "logs",
        ]:
            self.subdir(name)
        return self.root
