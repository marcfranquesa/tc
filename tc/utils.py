import PIL


def load_models():
    from ._ram import get_ram_plus_model
    from .sam import get_sam3_processor

    get_sam3_processor()
    get_ram_plus_model()


def boxes_overlap(
    box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
) -> bool:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # check if boxes don't overlap (easier to check)
    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    return True


def add_boxes(
    image: PIL.Image.Image,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
) -> PIL.Image.Image:
    if labels is None:
        labels = [""] * len(boxes)
    if colors is None:
        colors = ["red"] * len(boxes)
    
    assert len(boxes) == len(labels), "boxes and labels must have the same length"
    assert len(boxes) == len(colors), "boxes and colors must have the same length"

    image_with_boxes = image.convert("RGB")
    draw = PIL.ImageDraw.Draw(image_with_boxes)
    font = PIL.ImageFont.load_default(size=50)

    for box, label, color in zip(boxes, labels, colors):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = box[0]
        label_y = max(0, box[1] - text_height - 5)

        draw.rectangle(
            [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
            fill=color,
            outline=color,
        )

        # Draw text
        draw.text((label_x + 2, label_y - 10), label, fill=(255, 255, 255), font=font)
    return image_with_boxes
