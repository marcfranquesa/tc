import time

import PIL
import sam3
import torch
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    InferenceMetadata,
)
from sam3.train.data.sam3_image_dataset import Image as SAMImage
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from . import utils

_SAM3_MODEL = None
_SAM3_PROCESSOR = None


def get_sam3_model():
    global _SAM3_MODEL
    if _SAM3_MODEL is None:
        _SAM3_MODEL = sam3.build_sam3_image_model()
    return _SAM3_MODEL


def get_sam3_processor():
    global _SAM3_PROCESSOR
    if _SAM3_PROCESSOR is None:
        _SAM3_PROCESSOR = Sam3Processor(get_sam3_model())
    return _SAM3_PROCESSOR


def _get_processor():
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _get_postprocessor():
    return PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,
    )


def run_sam3(image: PIL.Image.Image, prompt: str) -> dict:
    processor = get_sam3_processor()

    start_time = time.time()
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    end_time = time.time()

    print(f"Time taken for inference: {end_time - start_time} seconds")
    return output


def run_sam3_batch(image: PIL.Image.Image, prompt: list[str]) -> dict:
    w, h = image.size

    images = [SAMImage(data=image, objects=[], size=[h, w])]
    find_queries = [
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=i,
                original_image_id=i,
                original_category_id=1,
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            ),
        )
        for i, text_query in enumerate(prompt)
    ]
    batch = Datapoint(find_queries=find_queries, images=images)
    batch = _get_processor()(batch)
    collated_batch = collate_fn_api([batch], dict_key="dummy")["dummy"]
    collated_batch = copy_data_to_device(
        collated_batch, torch.device("cuda"), non_blocking=True
    )

    model = get_sam3_model()
    output = model(collated_batch)

    postprocessor = _get_postprocessor()
    return postprocessor.process_results(output, collated_batch.find_metadatas)


def add_boxes_from_sam3(
    image: PIL.Image.Image, sam3_output: dict, prompt: list[str] | None = None
) -> PIL.Image.Image:
    if prompt is None:
        prompt = [""] * len(sam3_output)
    boxes = []
    _prompt = []
    for output, label in zip(sam3_output.values(), prompt):
        _boxes = output["boxes"].numpy()
        boxes.extend(_boxes)
        _prompt.extend([label] * len(_boxes))
    return utils.add_boxes(image, boxes, _prompt)
