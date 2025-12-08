
<p align="center">
  <img src="tensorleap_logo_rgb_blue.png" alt="Tensorleap logo" height="70" style="margin-right:24px;"/>
  <img src="assets/logo.png" alt="YOLOX logo" height="70"/>
</p>

This repository is a fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). This contains the enhancements of the YOLOX repository for supporting additional tasks and embedded-friendly ti_lite models. 


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
### Run Tensorleap Integration Test
Sanity check to see that the installation was successful, run:
```
python integration_test.py
```
## Personalize Project
Customizing the project to your needs requires you to:
1. Export an ONNX version of your trained model 
2. Add data to the Tensorleap-configured folder 
3. Edit tensorleap_config.yaml with your variables

### 1.  🚀 ONNX export

After installation, you can export a YOLOX ONNX using `tools/export_onnx.py`:
```
 python -m tools.export_onnx \
  -f exps/default/yolox_s_ti_lite.py \
  -c PATH-TO-PRE-TRAINED-WEIGHTS.PTH \                                                    
  --export-raw-head-with-det \
  --output-name yolox_s_raw_head_det_person.onnx \
  --num-classes NUMBER-OF-OUT-CLASSES
```
Key flags:
- `-f/--exp_file`: experiment definition; choose one of the exps in `exps/default/` or any other exp you use.
- `-c/--ckpt`: checkpoint to export, e.g. `pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth`.
- `--export-raw-head-with-det`: outputs both raw head tensors and post-NMS detections in one ONNX.
- `--output-name`: target ONNX filename.
- `--dataset`: set to `coco-person` to use 1 COCO label ('0' for person).

Outputs land in the current directory. Ensure the `pretrained_models` folder is present.

### 2.  📂 COCO-style dataset layout
Place your data in the Tensorleap data folder (point `DATA_ROOT` to your path) using the standard structure:

```
data/
  coco-style/
    annotations/
      instances_train.json
      instances_val.json
      unlabeled.json
    images/
        train/          # images
        val/            # images
        unlabeled/       # images
```
Custom subsets should keep the same `annotations/` + split-folder layout; update `DATA_ROOT` and annotation filenames accordingly in your exp or config.

### 3. 🔗 Tensorleap integration files
Update `tensorleap_config.yaml` → Run `integration_test.py` → Upload. <br> 
To push the project with your data, update the variables in `tensorleap_config.yaml`.
In the config file you will find:
- `CLASSES`: list of classes as defined in the dataset json file.
- `DATA_ROOT`: relative or absolute root to your data 
- `TRAIN/VAL/UNLABELED_JSON`: name of the json file (considering a DATA_ROOT / "annotations" / cfg["..._JSON"] data structure)
- `TRAIN/VAL/UNLABELES_NAME`: name of the folder (considering a DATA_ROOT / "images" / cfg["..._NAME"] data structure)
- `STRIDES`: number of resolution steps taken by the network (no need to change if using standard YOLOX versions)

After you update the config file, validate using: 
```
python integration_test.py
```
There are 2 CLI switches to integration_test.py:
- `--vis-results`: boolean flag to toggle visualization if you want to review.
- `--num-images`: integer to control how many samples run (max 10).

When everything runs smoothly you can run with the path to your model:

```
leap push PATH-TO-ONNX-MODEL
```

Review of key files: 
- `leap.yaml`: Tensorleap manifest pointing to `integration_test.py` as entry, Python 3.9, and excludes large/binary assets (onnx, pth, outputs, images). <br>
- Controls what gets uploaded to Tensorleap.
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
