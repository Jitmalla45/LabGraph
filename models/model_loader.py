import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2VLForConditionalGeneration


def torch_dtype(dtype_name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name, torch.float32)


def load_processor(base_model):
    return AutoProcessor.from_pretrained(base_model, trust_remote_code=True)


def load_model(base_model, lora_path, dtype="float32", device="cpu"):
    dtype = torch_dtype(dtype)
    try:
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device,
        )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model


def load_qwen_stack(config):
    processor = load_processor(config.base_model)
    model = load_model(config.base_model, config.lora_path, config.dtype, config.device)
    return model, processor
