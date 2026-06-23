from dataclasses import dataclass, field
from pathlib import Path

from utils.output_paths import OutputPathManager


@dataclass
class PipelineConfig:
    data_dir: Path = Path("Data")
    output_dir: Path = Path("outputs")
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    lora_path: str = "FINETUNED QWEN/physics_lora"
    device: str = "cpu"
    dtype: str = "float32"
    image_id: str = "9"
    max_spatial_relations: int = 2
    max_attributes: int = 2
    landmark_count: int = 3
    landmark_delta: int = 150
    iou_threshold: float = 0.7
    pixel_threshold: float = 25.0
    query_list: list[str] = field(default_factory=lambda: [
        "Where is the glasses with respect to battery?",
        "Where is the glasses?",
    ])

    @property
    def output_paths(self) -> OutputPathManager:
        return OutputPathManager(self.output_dir, self.image_id)

    @property
    def final_output_dir(self) -> Path:
        return self.output_paths.root
