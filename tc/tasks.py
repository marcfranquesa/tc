from PIL import Image
import numpy as np

from . import sam, utils


def _parse_prompt(prompt: str | list[str]):
    if isinstance(prompt, str):
        if prompt == "all":
            prompt = "item"
        prompt = [prompt]
    return prompt


def run_task1(image_path: str, prompt: str | list[str]):
    prompt = _parse_prompt(prompt)
    image = Image.open(image_path)
    output = sam.run_sam3_batch(image, prompt)
    image_with_boxes = sam.add_boxes_from_sam3(image, output, prompt)
    return image_with_boxes


def run_task2(image_path: str = "", prompt: str = ""):
    prompt = _parse_prompt(prompt) + ["barcode"]
    image = Image.open(image_path)
    output = sam.run_sam3_batch(image, prompt)

    results = list(output.values())
    barcodes = results[-1]
    bboxes = np.concatenate([result["boxes"].numpy() for result in results[:-1]], axis=0)
    bboxes_barcode = barcodes["boxes"].numpy() 

    overlapping_barcode_bboxes = []

    for barcode_bbox in bboxes_barcode:
        for other_bbox in bboxes:
            if utils.boxes_overlap(barcode_bbox, other_bbox):
                overlapping_barcode_bboxes.append(barcode_bbox)
                break

    image_with_boxes = utils.add_boxes(image, overlapping_barcode_bboxes)
    return output,image_with_boxes


def run_task3(image_path: str = "", prompt: str = ""):
    pass
