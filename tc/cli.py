from pathlib import Path

from fire import Fire

from . import tasks


def main(task: int, output_dir: str = "out", **kwargs):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == 1:
        image_path = kwargs.get("image_path", "")
        prompt = kwargs.get("prompt", "")
        output = tasks.run_task1(image_path=image_path, prompt=prompt)

        image_name = Path(image_path).stem
        output.save(out_dir / f"{image_name}-task1.png")

    elif task == 2:
        print("Hello from sam 2!")
    elif task == 3:
        print("Hello from sam 3!")
    else:
        print("Invalid task!")


def cli() -> None:
    Fire(main)
