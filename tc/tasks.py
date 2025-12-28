import numpy as np
import PIL

from . import _ram, barcodes, marigold, sam, utils


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
    barcode_results = results[-1]
    bboxes = np.concatenate(
        [result["boxes"].numpy() for result in results[:-1]], axis=0
    )
    barcode_bboxes = barcode_results["boxes"].numpy()
    barcode_masks = barcode_results["masks"].numpy()

    overlapping_barcode_bboxes = []
    overlapping_barcode_masks = []

    for barcode_bbox, barcode_mask in zip(barcode_bboxes, barcode_masks):
        for other_bbox in bboxes:
            if utils.boxes_overlap(barcode_bbox, other_bbox):
                overlapping_barcode_bboxes.append(barcode_bbox)
                overlapping_barcode_masks.append(barcode_mask.squeeze(0))
                break

    decoded_barcodes = barcodes.decode(image, overlapping_barcode_masks)
    image_with_boxes = utils.add_boxes(
        image, overlapping_barcode_bboxes, decoded_barcodes
    )
    normal_vectors = marigold.get_normals(image_path)
    image_with_boxes_and_normals = barcodes.add_normal_vectors(
        image_with_boxes, normal_vectors, overlapping_barcode_masks
    )
    return image_with_boxes_and_normals


def run_task3(image_path: str = "", prompt: str = ""):
    image = PIL.Image.open(image_path)
    sam3_prompt = ["item", "code128 barcode", prompt]
    sam3_output = sam.run_sam3_batch(image, sam3_prompt)
    items_bboxes = sam3_output[0]["boxes"].numpy()
    barcodes_bboxes = sam3_output[1]["boxes"].numpy()
    barcodes_masks = sam3_output[1]["masks"].numpy()
    prompt_bboxes = sam3_output[2]["boxes"].numpy()

    items_labels = _ram.label_boxes(image, items_bboxes)

    decoded_barcodes = barcodes.decode(image, barcodes_masks)

    requested_barcodes = []
    for decoded_barcode, barcode_bbox in zip(decoded_barcodes, barcodes_bboxes):
        if prompt == decoded_barcode:
            requested_barcodes.append(barcode_bbox)

    matched_items = [
        (
            bbox,
            [
                (item_bbox, label)
                for item_bbox, label in zip(items_bboxes, items_labels)
                if utils.boxes_overlap(bbox, item_bbox)
            ],
        )
        for bbox in requested_barcodes
    ]
    matched_barcodes = [
        (
            bbox,
            [
                (barcode_bbox, label)
                for barcode_bbox, label in zip(barcodes_bboxes, decoded_barcodes)
                if utils.boxes_overlap(bbox, barcode_bbox)
            ],
        )
        for bbox in prompt_bboxes
    ]
    task3_barcode_bboxes = [barcode[0] for barcode in matched_items]
    task3_item_matches_bboxes = [
        item[0] for barcode in matched_items for item in barcode[1]
    ]
    task3_item_matches_labels = [
        ",".join(item[1]) for barcode in matched_items for item in barcode[1]
    ]

    task3_item_bboxes = [item[0] for item in matched_barcodes]
    task3_barcode_matches_bboxes = [
        barcode[0] for item in matched_barcodes for barcode in item[1]
    ]
    task3_barcode_matches_labels = [
        barcode[1] for item in matched_barcodes for barcode in item[1]
    ]
    bboxes_matched_items = task3_item_bboxes + task3_barcode_matches_bboxes
    labels_matched_items = ["detected item"] * len(
        task3_item_bboxes
    ) + task3_barcode_matches_labels
    matched_barcodes = utils.add_boxes(
        image, bboxes_matched_items, labels_matched_items
    )

    bboxes_matched_barcodes = task3_barcode_bboxes + task3_item_matches_bboxes
    labels_matched_barcodes = ["detected barcode"] * len(
        task3_barcode_bboxes
    ) + task3_item_matches_labels
    matched_items = utils.add_boxes(
        image, bboxes_matched_barcodes, labels_matched_barcodes
    )

    return matched_items, matched_barcodes
