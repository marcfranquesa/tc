## TC: vision tasks with SAM3 + barcode decoding + normals

This repo implements three small computer-vision tasks over photos in `images/`.
The core building blocks are:

- **SAM3**: text-prompted detection/segmentation to get boxes + masks.
- **Code128 decoding**: manual decoder to turn barcode regions into strings.
- **Marigold normals**: estimates a surface normal field; we overlay one normal vector per detected barcode.
- **RAM+** (Recognize Anything Model): tags cropped item regions (used in Task 3).

Main entrypoints:

- **Python API**: `tc.run_task1`, `tc.run_task2`, `tc.run_task3`
- **CLI**: `cli` and `batch`
- **Notebook**: `demo.ipynb`

### How each task is implemented

All tasks are implemented in `tc/tasks.py`.

#### Task 1: prompted object detection (boxes)

- **Input**: image + `prompt` (`"all"` or list like `["bottle","tool"]`)
- **Implementation**:
  - `_parse_prompt()` converts `"all"` → `"item"` (easier for SAM3) and normalizes strings → list
  - `sam.run_sam3_batch(image, prompt)` runs SAM3 once (in batch, extremely fast) for all prompts
  - `sam.add_boxes_from_sam3(...)` overlays all predicted boxes with their prompt label
- **Output**: one annotated image (boxes for each prompt)

#### Task 2: barcodes on items (+ decode + normals)

- **Goal**: detect items and Code128 barcodes, keep only barcodes that overlap an item, decode them, and draw a surface-normal vector per barcode.
- **Implementation**:
  - Builds the SAM3 prompt list: `prompt + ["code128 barcode"]`
  - Runs `sam.run_sam3_batch(...)` and splits results into:
    - item boxes (from the user prompt)
    - barcode boxes + masks (from `"code128 barcode"`)
  - Filters to overlapping barcodes via `utils.boxes_overlap(...)`
  - Decodes strings with `tc.barcodes.decode(image, masks)`:
    - warps each masked region with `warp_from_mask(...)`
    - runs `decode_code128_best_effort(...)`
    - returns `"?"` on failure
  - Draws decoded strings using `utils.add_boxes(...)`
  - Computes normals with `marigold.get_normals(image_path)` and overlays a vector per mask via `barcodes.add_normal_vectors(...)` (mean normal over the mask, normalized, drawn as a red line).
- **Output**: one annotated image (decoded barcode labels + normal vectors)

#### Task 3: match “thing” to barcodes (two views)

Task 3 returns two annotated images:

- **Items that contain barcodes encoding `prompt`**
- **Barcodes that are on the item(s) detected by `prompt`**

Implementation details:

- Runs SAM3 with prompts: `["item", "code128 barcode", prompt]`
- Uses RAM+ to tag each detected item crop: `_ram.label_boxes(image, items_bboxes)`
- Decodes all barcode masks with `barcodes.decode(...)`
- **Barcode → item matching**:
  - selects barcodes whose decoded string equals `prompt`
  - for each such barcode bbox, collects overlapping item bboxes (`utils.boxes_overlap`)
  - draws the barcode bbox + matched item bbox(es) (labels are the RAM+ tag lists joined by commas)
- **Item(prompt) → barcode matching**:
  - for each bbox detected by SAM3 for `prompt`, collects overlapping decoded barcode bboxes
  - draws the prompt bbox + matched barcode bbox(es) (labels are the decoded strings)

### Setup

This project is configured with **Pixi**:

```bash
pixi install
pixi run python -c "import tc; tc.load_models(); print('models loaded')"
```

### Running

#### Notebook

`demo.ipynb` shows the end-to-end flow:

- `import tc`
- `tc.load_models()`
- `tc.run_task1/2/3(...)`

#### CLI (single run)

The CLI is exposed via `[project.scripts]` in `pyproject.toml`:

```bash
# Task 1 (prompt can be "all" or a list literal string)
pixi run cli --task 1 --image_path images/input_1.jpg --prompt "all"
pixi run cli --task 1 --image_path images/input_1.jpg --prompt '["bottle","tool","box"]'

# Task 2
pixi run cli --task 2 --image_path images/input_1.jpg --prompt "all"

# Task 3 (prompt is compared to decoded barcode text)
pixi run cli --task 3 --image_path images/input_1.jpg --prompt "bottle"
```

Outputs are written to `out/task{N}/...` by default (override with `--output_dir`).

#### Batch runner (YAML)

Run all examples in `examples.yaml`:

```bash
pixi run batch --yaml_path examples.yaml
```


### Limitations

- **Barcode decoding is the main bottleneck**: compared to SAM3 inference, decoding can be noticeably slower because it runs per detected barcode region and may require multiple attempts (warping/thresholding/rotations).
- **Decoding reliability**: the current decoder is not perfectly robust and may return `"?"` on difficult cases (blur, glare, low resolution, partial occlusion, curved surfaces).
- **Potential improvements**:
  - Increase image quality before decoding
  - Stronger warping/rectification