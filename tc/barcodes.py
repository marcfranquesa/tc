import PIL
import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np


def _estimate_normal_vector(barcode_mask: np.ndarray) -> np.ndarray:
    # get the bounding box of the barcode
    x, y, w, h = cv2.boundingRect(barcode_mask)
    # get the center of the barcode
    center_x = x + w / 2
    center_y = y + h / 2
    # get the normal vector
    normal_vector = np.array([center_x, center_y])
    return normal_vector


def estimate_normal_vectors(barcode_masks: np.ndarray) -> np.ndarray:
    normal_vectors = []
    for barcode_mask in barcode_masks:
        normal_vector = _estimate_normal_vector(barcode_mask)
        normal_vectors.append(normal_vector)
    return np.array(normal_vectors)


def decode(
    image: PIL.Image.Image, bboxes: list[tuple[float, float, float, float]]
) -> list[str]:
    decoded_barcodes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        cropped_image = image.crop((x, y, x + w, y + h))
        decoded = pyzbar.decode(cropped_image)
        if decoded:
            decoded_barcodes.append(decoded[0].data.decode("utf-8"))
        else:
            decoded_barcodes.append("?")
    return decoded_barcodes
