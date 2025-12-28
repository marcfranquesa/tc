from __future__ import annotations

from typing import Any, List

import numpy as np
from PIL import Image

from .decode_barcodes import decode_code128_best_effort, warp_from_mask


def decode(image: Image.Image, masks: List[Any]) -> List[str]:
    """
    Decode Code128 barcodes from an image given per-barcode masks.

    Returns a list of decoded strings aligned with `masks`.
    If a mask cannot be decoded as Code128, returns '?' for that entry.
    """
    rgb = np.array(image.convert("RGB"))
    bgr = rgb[:, :, ::-1].copy()

    out: List[str] = []
    for m in masks:
        if m.ndim == 3:
            m = m.squeeze(0)
        mask = np.asarray(m)
        if mask.dtype != np.bool_:
            mask = mask.astype(np.uint8) > 0

        try:
            view = warp_from_mask(bgr, mask, scale_w=1.6, scale_h=1.2, out_scale=9.0)
            res, _, _ = decode_code128_best_effort(view)
            out.append(res.text if res is not None and res.text else "?")
        except Exception:
            out.append("?")

    return out
