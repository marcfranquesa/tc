import PIL


def load_models():
    from .sam import get_sam3_processor

    get_sam3_processor()

def boxes_overlap(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> bool:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # check if boxes don't overlap (easier to check)
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def add_boxes(
    image: PIL.Image.Image, boxes: list[tuple[int, int, int, int]], labels: list[str] | None = None
) -> PIL.Image.Image:
    if labels is None:
        labels = [""] * len(boxes)

    image_with_boxes = image.convert("RGB")
    draw = PIL.ImageDraw.Draw(image_with_boxes)
    font = PIL.ImageFont.load_default(size=60)

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin + 50, ymax - 100), label, fill="red", font=font)
    return image_with_boxes
