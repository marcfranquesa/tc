import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def add_masks_from_sam3(image: Image.Image, sam3_output: dict) -> dict:
    masks = sam3_output["masks"]
    masks_np = masks.detach().cpu().numpy()[:, 0, ...].astype(np.uint8)

    base = image.convert("RGBA")
    W, H = base.size

    for i, mask in enumerate(masks_np):
        if mask.shape != (H, W):
            mask = np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
                    (W, H), resample=Image.Resampling.NEAREST
                )
            )
            mask = (mask > 0).astype(np.uint8)

        r, g, b, _ = plt.cm.tab10(i % 10)
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[..., 0] = int(r * 255)
        overlay[..., 1] = int(g * 255)
        overlay[..., 2] = int(b * 255)
        overlay[..., 3] = (mask * 128).astype(np.uint8)  # ~0.5 alpha

        base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    image_with_masks = base.convert("RGB")
    return image_with_masks
