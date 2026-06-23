from pathlib import Path

from PIL import ImageDraw, ImageFont

from utils.helpers import object_display_name


def landmark_grounding(image, target, reference, relation, save_path=None):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for obj, color, label in [
        (target, "lime", f"target: {object_display_name(target)}"),
        (reference, "red", f"reference: {object_display_name(reference)}"),
    ]:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(0, y1 - 22)), label, fill=color, font=font)

    text = f"{object_display_name(target)} is {relation.replace('_', ' ')} {object_display_name(reference)}"
    draw.text((10, 10), text, fill="yellow", font=font)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    return text
