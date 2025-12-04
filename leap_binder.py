import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_loader.utils import rescale_min_max
from sklearn.model_selection import train_test_split
from yolox.utils.visualize import vis as yolox_vis
from code_loader.contract.datasetclasses import DataStateType, PreprocessResponse, SamplePreprocessResponse
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox, LeapImage
from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader import leap_binder
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_metadata,
    tensorleap_custom_visualizer,
    tensorleap_custom_loss,
)

from yolox.data.datasets import COCO_CLASSES, COCODataset
from yolox.data.data_augment import ValTransform
from yolox.utils import bboxes_iou, PostprocessExport, postprocess
from yolox.models.losses import IOUloss


COCO_ROOT = Path("/Users/orram/Tensorleap/data/coco-person128")
NUM_CLASSES = 1 #len(COCO_CLASSES)
COCO_ANN = COCO_ROOT / "annotations" / "instances_val2017_32.json"
STRIDES = [8, 16, 32]


@dataclass
class Sample:
    image_path: Path
    boxes: np.ndarray  # shape (N, 5) -> [x0, y0, x1, y1, class_id]
    filename: str
    image_id: int


def _load_coco(name="val") -> COCODataset:
    """Load coco-person128 split; fail loudly if missing."""
    has_coco = COCO_ANN.exists()
    if not has_coco:
        raise FileNotFoundError("coco-person128 subset not found. Run tools/make_coco_person128.py")

    if name == "val":
        ds_name = "val2017"
        json_file = "instances_val2017_32.json"
    elif name == "unlabeled":
        ds_name = "unlabeled2017"
        json_file = "image_info_unlabeled2017_50.json"
    elif name == "train":
        ds_name = "train2017"
        json_file = "instances_train2017_128.json"
    else:
        raise KeyError("Unknown dataset")

    dataset = COCODataset(
        data_dir=str(COCO_ROOT),
        json_file=json_file,
        name=ds_name,
        img_size=(640, 640),
        preproc=ValTransform(legacy=False, visualize=True),
    )
    return dataset


@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    train_data = _load_coco('train')
    val_data = _load_coco('val')
    unlabled_data = _load_coco('unlabeled')
    if len(train_data) == 0 or len(val_data) == 0:
        raise RuntimeError("No samples found in COCO dataset.")
    return [
        PreprocessResponse(length=len(train_data), data={"samples": train_data}, state=DataStateType.training),
        PreprocessResponse(length=len(val_data), data={"samples": val_data}, state=DataStateType.validation),
        PreprocessResponse(length=len(unlabled_data), data={"samples": unlabled_data}, state=DataStateType.unlabeled),
    ]


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data["samples"][idx]
    img, target, img_info, img_id = sample
    return img.astype(np.float32)


@tensorleap_gt_encoder("bboxes")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    sample = preprocess.data["samples"][idx]
    img, target, img_info, img_id = sample
    return target.astype(np.float32)


@tensorleap_metadata(
    name="metadata",
    metadata_type={
        "file_name": DatasetMetadataType.string,
        "image_id": DatasetMetadataType.int,
        "num_objects": DatasetMetadataType.int,
    },
)
def metadata_per_img(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    sample = preprocess.data["samples"][idx]
    img, target, img_info, img_id = sample
    return {"orig_H": img_info[0], f"orig_W": img_info[1]}

def post_process_image(image, meta_data):
    orig_H, orig_W = meta_data["orig_H"], meta_data["orig_W"]
    H, W = image.shape[-2:]
    # Image to uint8 HWC, undo padding
    img_viz = rescale_min_max(image.squeeze(0))
    img_viz = np.clip(img_viz, 0, 255).astype(np.uint8)
    img_viz = np.transpose(img_viz, axes=[1, 2, 0])
    padded_h, padded_w = img_viz.shape[:2]
    r = min(padded_h / orig_H, padded_w / orig_W)
    resized_h, resized_w = int(orig_H * r), int(orig_W * r)
    img_viz = img_viz[:resized_h, :resized_w]
    img_viz = cv2.resize(img_viz, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
    return img_viz, r

@tensorleap_custom_visualizer("image", LeapDataType.Image)
def image_visualizer(image: np.ndarray, data: SamplePreprocessResponse,
) -> LeapImage:
    meta_data = metadata_per_img(int(data.sample_ids), data.preprocess_response)
    # Convert model input back to displayable uint8 HWC and undo padding
    img_viz, r = post_process_image(image, meta_data)
    return LeapImage(img_viz, compress=False)

@tensorleap_custom_visualizer("image_with_boxes", LeapDataType.Image)
def image_with_boxes_visualizer(
    image: np.ndarray,
    bboxes: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImage:
    meta_data = metadata_per_img(int(data.sample_ids), data.preprocess_response)
    # Convert model input back to displayable uint8 HWC and undo padding
    img_viz, r = post_process_image(image, meta_data)

    bboxes = bboxes.squeeze(0)
    boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in bboxes], dtype=np.float32)
    cls_ids = np.array([b[4] for b in bboxes], dtype=np.int32)
    scores = np.ones(len(boxes_arr), dtype=np.float32)
    img_viz = yolox_vis(img_viz.copy(), boxes_arr, scores, cls_ids, conf=0.0, class_names=COCO_CLASSES)

    return LeapImage(img_viz, compress=False)


@tensorleap_custom_visualizer("image_with_pred_boxes", LeapDataType.Image)
def image_with_pred_boxes_visualizer(
    image: np.ndarray,
    preds: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImage:
    """
    Visualize predictions in (xyxy + obj + class scores) format from pre-NMS output.
    """
    if preds.ndim > 2:
        boxes = postprocess(torch.tensor(preds), conf_thre=0.3, nms_thre=0.45,
                        num_classes=NUM_CLASSES, class_agnostic=True)[0]
    else:
        boxes = preds
    meta_data = metadata_per_img(int(data.sample_ids), data.preprocess_response)
    img_viz, r = post_process_image(image, meta_data)
    if boxes is None or boxes.size == 0:
        return LeapImage(img_viz, compress=False)
    boxes = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
    cls_ids = np.array([b[-1] for b in boxes], dtype=np.int32)
    scores_obj = np.array([b[4] for b in boxes], dtype=np.float32)
    # denormalize
    boxes_arr[:, [0, 2]] /= r
    boxes_arr[:, [1, 3]] /= r
    cls_ids_np = cls_ids.astype(np.int32)
    img_viz = yolox_vis(img_viz.copy(), boxes_arr, scores_obj, cls_ids_np, conf=0.0, class_names=COCO_CLASSES )

    return LeapImage(img_viz, compress=False)



# --------------------------------------------------------------------------- #
# YOLOX head loss using raw head outputs (pre-decode)                         #
# --------------------------------------------------------------------------- #

def yolox_head_loss_raw(pred80, pred40, pred20, gt_bboxes: np.ndarray):
    """
    Compute YOLOX loss from raw head outputs (per-level tensors before decode/NMS).

    Expected ONNX outputs (from --export-raw-head):
      head_outs_0/1/2: [B, C, H, W] with C = 4 (reg) + 1 (obj) + num_classes (cls)
      strides: [S] holding stride per level (e.g., [8, 16, 32])

    gt_bboxes: [..., 5] with [x1, y1, x2, y2, class]
    Returns: scalar loss [[float32]]
    """

    from yolox.models.yolo_head import YOLOXHead
    head_outs = [pred80, pred40, pred20]
    # Normalize inputs
    if isinstance(head_outs, (list, tuple)):
        outs_np = [np.asarray(o) for o in head_outs]
    else:
        # ONNXRuntime may deliver a flat tuple; handle single-level too
        outs_np = [np.asarray(head_outs)]
    strides = np.asarray(STRIDES).flatten()
    gt_bboxes = np.asarray(gt_bboxes)

    if gt_bboxes.ndim == 3:
        gt_bboxes = gt_bboxes[0]
    # if len(outs_np) == 0 or gt_bboxes.size == 0:
    #     return np.array([0.0], dtype=np.float32)

    num_levels = len(outs_np)
    num_classes = outs_np[0].shape[1] - 5  # C = 4+1+num_classes

    # Build a lightweight head just to reuse get_output_and_grid / get_losses
    dummy_in_channels = [1] * num_levels
    head = YOLOXHead(num_classes=num_classes, in_channels=dummy_in_channels, strides=list(strides))
    head.decode_in_inference = False
    head.use_l1 = False

    outputs_decoded = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []

    dtype = torch.float32
    for k, (out_np, stride) in enumerate(zip(outs_np, strides)):
        out_t = torch.from_numpy(out_np).to(dtype)
        # get_output_and_grid expects packed channels [B, C, H, W]
        output, grid = head.get_output_and_grid(out_t, k, stride, out_t.type())
        outputs_decoded.append(output)
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(torch.full((1, grid.shape[1]), float(stride), dtype=dtype))

    outputs_cat = torch.cat(outputs_decoded, dim=1)  # [B, N, 5+num_classes]

    # Build labels tensor [B, max_gt, 5] with class-first and xywh
    gt_xyxy = torch.from_numpy(gt_bboxes[:, :4]).to(dtype)
    gt_cls = torch.from_numpy(gt_bboxes[:, -1:]).to(dtype)
    gt_cxcy = (gt_xyxy[:, 0:2] + gt_xyxy[:, 2:4]) / 2.0
    gt_wh = (gt_xyxy[:, 2:4] - gt_xyxy[:, 0:2]).clamp(min=1e-6)
    labels = torch.zeros((1, gt_xyxy.shape[0], 5), dtype=dtype)
    labels[0, :, 0:1] = gt_cls
    labels[0, :, 1:3] = gt_cxcy
    labels[0, :, 3:5] = gt_wh

    (loss, loss_iou, loss_obj,
     loss_cls, loss_l1, _) = head.get_losses(
        imgs=None,
        x_shifts=x_shifts,
        y_shifts=y_shifts,
        expanded_strides=expanded_strides,
        labels=labels,
        outputs=outputs_cat,
        origin_preds=None,
        dtype=dtype,
    )

    return loss.detach().cpu().numpy().astype(np.float32), {'loss_iou':loss_iou, 'loss_obj':loss_obj,
     'loss_cls':loss_cls, 'loss_l1':loss_l1}

@tensorleap_custom_loss(name="total_loss")
def total_loss(pred80, pred40, pred20, gt_bboxes: np.ndarray) -> np.ndarray:
    total_loss, _ = yolox_head_loss_raw(pred80, pred40, pred20, gt_bboxes)
    return total_loss

