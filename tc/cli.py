from pathlib import Path

from fire import Fire
import yaml

from . import tasks, utils

def _cli(task: int, image_path: str = "", prompt: str = "", output_dir: str = "out"):
    out_dir = Path(output_dir)
    _out_dir = out_dir / f"task{task}"
    _out_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    if task == 1:
        try:
            import ast

            parsed = ast.literal_eval(prompt)
            if isinstance(parsed, list):
                prompt = parsed
        except (ValueError, SyntaxError):
            pass
        output = tasks.run_task1(image_path=image_path, prompt=prompt)
        utils.save(output, path=f"{_out_dir}/{image_name}-{prompt}.png")
    elif task == 2:
        output = tasks.run_task2(image_path=image_path, prompt=prompt)
        utils.save(output, path=f"{_out_dir}/{image_name}-{prompt}.png")
    elif task == 3:
        output_items, output_barcodes = tasks.run_task3(
            image_path=image_path, prompt=prompt
        )
        utils.save(output_items, path=f"{_out_dir}/{image_name}-{prompt}-items.png")
        utils.save(
            output_barcodes, path=f"{_out_dir}/{image_name}-{prompt}-barcodes.png"
        )
    else:
        print("Invalid task!")


def cli() -> None:
    Fire(_cli)


def _batch(yaml_path: str = "examples.yaml"):
    with open(yaml_path, "r") as f:
        tasks = yaml.safe_load(f)

    for task_name, task_data in tasks.items():
        _cli(task_data["task"], task_data["image_path"], task_data["prompt"])


def batch():
    Fire(_batch)
