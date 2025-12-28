import numpy as np
import PIL
import pyzbar.pyzbar as pyzbar


def add_normal_vectors(
    image: PIL.Image.Image, marigold_normals: np.ndarray, barcode_masks: np.ndarray
) -> PIL.Image.Image:
    image_with_normals = image.convert("RGB").copy()
    draw = PIL.ImageDraw.Draw(image_with_normals)

    line_len = 200
    line_w = 6

    for mask in barcode_masks:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())

        # average + normalize normal within the masked area
        v = np.nanmean(marigold_normals[ys, xs, :], axis=0)
        if not np.all(np.isfinite(v)):
            continue

        nrm = float(np.linalg.norm(v))
        v = (v / nrm).astype(np.float32)

        if v[2] < 0:
            v = -v

        x0, y0 = int(round(cx)), int(round(cy))
        x1 = int(round(cx + float(v[0]) * line_len))
        y1 = int(round(cy - float(v[1]) * line_len))
        draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=line_w)

    return image_with_normals


def decode_pyzbar(
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
