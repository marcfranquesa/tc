
from PIL import Image

from . import utils
from .sam import run_sam3


def run_task1(image_path: str = "", prompt: str = ""):
    if prompt == "all":
        prompt = "items"
    image = Image.open(image_path)
    output = run_sam3(image, prompt)
    image_with_masks = utils.add_masks_from_sam3(image, output)

    return image_with_masks
