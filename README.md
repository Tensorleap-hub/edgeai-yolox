
<p align="center">
  <img src="tensorleap_logo_rgb_blue.png" alt="Tensorleap logo" height="70" style="margin-right:24px;"/>
  <img src="assets/logo.png" alt="YOLOX logo" height="70"/>
</p>

This repository is a fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). This contains the enhancements of the YOLOX repository for supporting additional tasks and embedded friendly ti_lite models. 


### Installation

#### Step0. Install conda (if not already installed)
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda and make sure `conda` is on your `PATH` (restart the shell if needed).

#### Step1. Install YOLOX (creates/uses a conda env)
`setup.sh` now creates/activates a conda environment named `edgeai-yolox` with Python 3.9 by default, then installs CUDA 11.8 compatible PyTorch
```
./setup.sh
```
Environment defaults can be overridden, e.g. `ENV_NAME=myenv PYTHON_VERSION=3.9 ./setup.sh`.

#### Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi) (if not already present)
`setup.sh` installs `pycocotools` via pip; if you need to reinstall manually:
```
conda activate edgeai-yolox
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### ONNX export

After installation, you can export a YOLOX ONNX for COCO using `tools/export_onnx.py`:
```
 python -m tools.export_onnx \
  -f exps/default/yolox_s_ti_lite.py \
  -c PATH-TO-PRE-TRAINED-WEIGHTS.PTH \                                                    
  --export-raw-head-with-det \
  --output-name yolox_s_raw_head_det_person.onnx \
  --dataset coco-persons
```

Key flags:
- `-f/--exp_file`: experiment definition; choose of the exp in  `exps/default/` or any other exp you used.
- `-c/--ckpt`: checkpoint to export, e.g. `pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth`.
- `--export-raw-head-with-det`: outputs both raw head tensors and post-NMS detections in one ONNX.
- `--output-name`: target ONNX filename.
- `--dataset`: set to `coco-person` to use 1 COCO label ('0' for person).

Other useful options:
- `--export-det` (decoded detections only), `--export-pre-nms` (add pre-NMS preds), `--export-raw-head` (raw heads only).
- `--dynamic` (dynamic input shape), `--batch-size`, `--opset` (default 11), `--no-onnxsim` to skip onnx-simplifier.
- `--task` for pose/human pose variants; `--train_ann/--val_ann` to point at custom COCO-format jsons.

Outputs land in the current directory (ONNX plus optional prototxt when TIDL metadata is produced). Ensure the `pretrained_models` folder is present and the COCO dataset path matches your exp file.

### Tensorleap integration files
- `leap.yaml`: Tensorleap manifest pointing to `integration_test.py` as entry, Python 3.9, and excludes large/binary assets (onnx, pth, outputs, images). Controls what gets uploaded to Tensorleap.
- `leap_binder.py`: Wiring for Tensorleap data/model bindings. It loads the `coco-person128` subset via `COCODataset`, defines preprocess responses, input/GT encoders, metadata, visualizers, and a raw-head YOLOX loss used with ONNX exports that include raw heads.
- `integration_test.py`: Entry declared in `leap.yaml`. Loads an ONNX model via `onnxruntime`, runs samples from the binders, computes the raw-head loss, and renders visualizations—serves as a local smoke test and Tensorleap integration check. The default ONNX path aligns with the export command above (`yolox_s_raw_head_det_1.onnx`).

More detail for Tensorleap users:
- `leap.yaml` — the manifest Tensorleap reads on upload. It pins the entry point (`integration_test.py`), Python 3.9 runtime, and excludes heavy artifacts (onnx/pth/outputs/images/docs) so only the code needed for execution is shipped. Update this if you change the entry script or want to include extra assets.
- `leap_binder.py` — all the Tensorleap “binders”:
  - `@tensorleap_preprocess` builds three splits from the local `coco-person128` subset (train/val/unlabeled), returning `PreprocessResponse` objects Tensorleap iterates over.
  - `@tensorleap_input_encoder` and `@tensorleap_gt_encoder` map an index to the model input tensor and ground-truth boxes.
  - `@tensorleap_metadata` plus custom visualizers turn inputs/labels/preds into images for the UI.
  - `@tensorleap_custom_loss` (`yolox_head_loss_raw`) computes YOLOX loss from raw head outputs—useful when exporting ONNX with `--export-raw-head-with-det`.
- `integration_test.py` — declared as the entry in `leap.yaml`. It loads an ONNX model via onnxruntime, pulls samples through the binders, computes the raw-head loss, and renders visualizations. The default model path matches the provided export command (`yolox_s_raw_head_det_1.onnx`). Running this locally is a quick smoke test; on Tensorleap it validates the integration.
- `data/coco-person128` — a trimmed COCO subset (128 person images) is bundled for quick smoke tests and Tensorleap demos; it keeps downloads light while matching the `coco-person` label set used in the examples.



### Note:
See the [original documentation](README_megvii.md)
