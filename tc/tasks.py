import numpy as np
import PIL

from . import sam, utils, _ram, barcodes


def _parse_prompt(prompt: str | list[str]):
    if isinstance(prompt, str):
        if prompt == "all":
            prompt = "item"
        prompt = [prompt]
    return prompt


def run_task1(image_path: str, prompt: str | list[str]):
    prompt = _parse_prompt(prompt)
    image = PIL.Image.open(image_path)
    output = sam.run_sam3_batch(image, prompt)
    image_with_boxes = sam.add_boxes_from_sam3(image, output, prompt)
    return image_with_boxes


def run_task2(image_path: str = "", prompt: str = ""):
    prompt = _parse_prompt(prompt) + ["code128 barcode"]
    image = PIL.Image.open(image_path)
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
    return (
        output,
        image_with_boxes,
        overlapping_barcode_bboxes,
        overlapping_barcode_masks,
    )


def run_task3(image_path: str = "", prompt: str = ""):
    image = PIL.Image.open(image_path)
    sam3_prompt = ["item", "code128 barcode", prompt]
    sam3_output = sam.run_sam3_batch(image, sam3_prompt)
    items_bboxes = sam3_output[0]["boxes"].numpy()
    barcodes_bboxes = sam3_output[1]["boxes"].numpy()
    prompt_bboxes = sam3_output[2]["boxes"].numpy()
    
    items_labels = _ram.label_boxes(image, items_bboxes)

    decoded_barcodes = barcodes.decode(image, barcodes_bboxes)

    requested_barcodes = []
    for decoded_barcode, barcode_bbox in zip(decoded_barcodes, barcodes_bboxes):
        if prompt == decoded_barcode:
            requested_barcodes.append(barcode_bbox)

    matched_items = [
        (bbox, [
            (item_bbox, label)
            for item_bbox, label in zip(items_bboxes, items_labels)
            if utils.boxes_overlap(bbox, item_bbox)
        ])
        for bbox in requested_barcodes
    ]
    matched_barcodes = [
        (bbox, [
            (barcode_bbox, label)
            for barcode_bbox, label in zip(barcodes_bboxes, decoded_barcodes)
            if utils.boxes_overlap(bbox, barcode_bbox)
        ])
        for bbox in prompt_bboxes
    ]
    return matched_items, matched_barcodes
