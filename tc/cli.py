from pathlib import Path

from fire import Fire

from . import tasks


def main(task: int, output_dir: str = "out", **kwargs):
    out_dir = Path(output_dir)
    _out_dir = out_dir / f"task{task}"
    _out_dir.mkdir(parents=True, exist_ok=True)

    image_path = kwargs.get("image_path", "")
    image_name = Path(image_path).stem
    prompt = kwargs.get("prompt", "")

    if task == 1:
        try:
            import ast

            parsed = ast.literal_eval(prompt)
            if isinstance(parsed, list):
                prompt = parsed
        except (ValueError, SyntaxError):
            pass
        output = tasks.run_task1(image_path=image_path, prompt=prompt)
    elif task == 2:
        output = tasks.run_task2(image_path=image_path, prompt=prompt)
    elif task == 3:
        print("Hello from sam 3!")
    else:
        print("Invalid task!")

    output.save(_out_dir / f"{image_name}-{prompt}.png")


def cli() -> None:
    Fire(main)
