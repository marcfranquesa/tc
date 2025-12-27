import PIL
from ram.models import ram_plus
import ram
import torch
from huggingface_hub import hf_hub_download

_RAM_PLUS_MODEL = None
_WEIGHTS_PATH = hf_hub_download(
    repo_id="xinyu1205/recognize-anything-plus-model",
    filename="ram_plus_swin_large_14m.pth",
)
_IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_ram_plus_model():
    global _RAM_PLUS_MODEL
    if _RAM_PLUS_MODEL is None:
        _RAM_PLUS_MODEL = ram_plus(
            pretrained=str(_WEIGHTS_PATH), image_size=_IMAGE_SIZE, vit="swin_l"
        )
        _RAM_PLUS_MODEL.eval()
        _RAM_PLUS_MODEL.to(DEVICE)
    return _RAM_PLUS_MODEL


def label_boxes(image: PIL.Image.Image, bboxes: list[tuple[int, int, int, int]]):
    model = get_ram_plus_model()
    transform = ram.get_transform(_IMAGE_SIZE)

    raw_image = image.convert("RGB")
    results = []

    for box in bboxes:
        crop = raw_image.crop((box[0], box[1], box[2], box[3]))
        image_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        res = ram.inference_ram(image_tensor, model)

        all_tags = res[0].split(" | ")

        results.append(all_tags)

    return results
