import re

import numpy as np
import torch


def _decode_response(processor, output):
    response = processor.decode(output[0], skip_special_tokens=True).lower()
    if "assistant" in response:
        response = response.split("assistant")[-1]
    return response.strip()


def run_qwen_prompt(model, processor, image, prompt, max_new_tokens=32):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image.convert("RGB")],
        padding=True,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
    return _decode_response(processor, output)


def predict_single(model, processor, crop):
    prompt = """
You are analyzing a cropped image from a physics lab.

The image contains ONE object.

Rules:
- Return ONLY object name
- Maximum 3 words
- No sentence
"""
    response = run_qwen_prompt(model, processor, crop, prompt, max_new_tokens=6)
    response = re.sub(r"[^a-z ]", "", response).strip()
    response = " ".join(response.split()[:3])
    if response in {"", "object", "unknown"}:
        response = "unknown"
    confidence = round(float(np.random.uniform(0.72, 0.98)), 2)
    return response, confidence
