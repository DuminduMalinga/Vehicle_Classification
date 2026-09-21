# Vehicle Classification (YOLOv8)

Object detection for four vehicle classes — **car, bus, truck, motorcycle** — using
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), fine-tuned from the
`yolov8n` (nano) checkpoint.

## Problem

Given an image or video frame, detect and classify vehicles into one of four categories:

| id | class      |
|----|------------|
| 0  | car        |
| 1  | bus        |
| 2  | truck      |
| 3  | motorcycle |

## Dataset

- ~3,000 images (2,100 train / 900 val) with per-object bounding boxes.
- Annotations were exported in **Supervisely JSON** format (`labels/{train,val}/ann/*.json`),
  one JSON file per image, with `objects[].classTitle` and `objects[].points.exterior`
  giving the box corners.
- The dataset itself is **not** committed to this repo (see [Reproducing](#reproducing)) —
  it lived under `data/vehicles/` locally and is excluded via `.gitignore`.

## Approach

1. **Convert annotations.** `Vehicle_classificationModel.ipynb` converts the Supervisely
   JSON boxes to YOLO's normalized `class x_center y_center width height` `.txt` format
   and writes a `data_yolo.yaml` pointing at the image folders.
2. **Train.** Fine-tune `yolov8n.pt` on the converted dataset with the Ultralytics
   `model.train(...)` API (`imgsz=640`, `batch=8`, CPU).
3. **Predict.** Run the trained weights on a sample image (`examples/input.jpg`) with
   `model.predict(...)` and save the annotated output.

## Known issue: label conversion path bug

The conversion code globs for `*.json` directly inside `labels/train/` and `labels/val/`,
but the actual per-image JSON files live one level deeper, in `labels/train/ann/` and
`labels/val/ann/`. As a result, the converter found **zero** JSON files and wrote
**zero** YOLO `.txt` labels, so both training runs (`runs/vehicles_yolov8`,
`runs/vehicles_yolov82`) trained on images with no ground-truth boxes.

This is visible directly in the logged metrics — `train/box_loss`, `train/dfl_loss`,
`metrics/precision`, `metrics/recall`, and `metrics/mAP50` are `0` for every epoch in
both `runs/*/results.csv`, and only the classification loss (`cls_loss`) moved.
The prediction demo in the notebook likewise fell back to the stock (COCO-pretrained)
`yolov8n.pt` weights rather than a fine-tuned checkpoint, because no `best.pt`/`last.pt`
existed yet at the time that cell ran.

**Fix:** point the converter at `labels/{train,val}/ann/*.json` instead of
`labels/{train,val}/*.json`, re-run the conversion cell so `.txt` files land next to the
images, then re-train. This has not been done yet — treat the current `runs/` artifacts
as a record of the bug, not as a trained model.

## Results

No valid accuracy metrics exist yet because of the bug above — box loss and mAP stayed
at `0` across both training runs. Once the label path is fixed and the model is
re-trained, replace this section with the resulting precision/recall/mAP50/mAP50-95
(available in `runs/<run_name>/results.csv` and `results.png` after training).

## Repo layout

```
Vehicle_classificationModel.ipynb   Annotation conversion + train + predict workflow
Vehicle Classification.ipynb        Earlier/exploratory notebook
data.yaml                           Minimal YOLO data config (paths are local; adjust before use)
examples/                           Sample images used for prediction demos
runs/                               Training/prediction outputs (plots, results.csv) — weights excluded, see below
```

## Model weights

Trained weights (`yolov8n.pt` base checkpoint, and `best.pt`/`last.pt` from each run
under `runs/*/weights/`) are **not** tracked in git — they're excluded via `.gitignore`
to keep the repository small. Given the label bug above, none of the currently-produced
weights are actually usable as a trained detector; once a valid run exists, its weights
should be attached to a [GitHub Release](../../releases) rather than committed.

## Reproducing

1. Install dependencies: `pip install ultralytics opencv-python pyyaml`.
2. Place the dataset under `data/vehicles/` with `images/{train,val}` and
   `labels/{train,val}/ann/*.json` (Supervisely format), matching the paths the
   notebook expects.
3. Open `Vehicle_classificationModel.ipynb` and run the cells top to bottom — update
   the hard-coded local paths (`PROJECT_DIR`, `IMAGE_PATH`) for your machine first.
4. Fix the annotation-conversion path (see [Known issue](#known-issue-label-conversion-path-bug))
   before trusting the training results.
