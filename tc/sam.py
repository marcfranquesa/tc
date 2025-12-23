import sam3
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor


def run_sam3(image: Image.Image, prompt: str) -> dict:
    model = sam3.build_sam3_image_model()
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    return output
