import numpy as np
from PIL import Image

from . import sam, utils, _ram


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
    prompt = _parse_prompt(prompt) + ["code128 barcode"]
    image = Image.open(image_path)
    output = sam.run_sam3_batch(image, prompt)

    results = list(output.values())
    barcodes = results[-1]
    bboxes = np.concatenate(
        [result["boxes"].numpy() for result in results[:-1]], axis=0
    )
    barcode_bboxes = barcodes["boxes"].numpy()
    barcode_masks = barcodes["masks"].numpy()

    overlapping_barcode_bboxes = []
    overlapping_barcode_masks = []

    for barcode_bbox, barcode_mask in zip(barcode_bboxes, barcode_masks):
        for other_bbox in bboxes:
            if utils.boxes_overlap(barcode_bbox, other_bbox):
                overlapping_barcode_bboxes.append(barcode_bbox)
                overlapping_barcode_masks.append(barcode_mask.squeeze(0))
                break

    image_with_boxes = utils.add_boxes(image, overlapping_barcode_bboxes)
    return output, image_with_boxes, overlapping_barcode_masks


def run_task3(image_path: str = "", prompt: str = ""):
    sam3_prompt = ["item"]
    sam3_output = sam.run_sam3_batch(image_path, sam3_prompt)
    sam3_bboxes = np.concatenate([result["boxes"].numpy() for result in sam3_output.values()], axis=0)
    labels = _ram.label_boxes(image_path, sam3_bboxes)
    print(labels)
