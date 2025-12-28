import diffusers
import torch

_MARIGOLD_NORMALS_PIPELINE = None


def get_marigold_normals_pipeline():
    global _MARIGOLD_NORMALS_PIPELINE
    if _MARIGOLD_NORMALS_PIPELINE is None:
        _MARIGOLD_NORMALS_PIPELINE = diffusers.MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16
        ).to("cuda")
    return _MARIGOLD_NORMALS_PIPELINE


def get_normals(image_path: str):
    pipeline = get_marigold_normals_pipeline()
    image = diffusers.utils.load_image(image_path)
    normals = pipeline(image)
    return normals.prediction.squeeze(0)
