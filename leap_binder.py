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

from code_loader.contract.datasetclasses import DataStateType, PreprocessResponse
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
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
from yolox.utils import bboxes_iou
from yolox.models.losses import IOUloss


COCO_ROOT = Path("/Users/orram/Tensorleap/data/coco")
COCO_ANN = COCO_ROOT / "annotations" / "instances_val2017.json"
COCO_IMG_DIR = COCO_ROOT / "images" / "val2017"
MAX_SAMPLES = 32
VAL_SPLIT = 0.2


@dataclass
class Sample:
    image_path: Path
    boxes: np.ndarray  # shape (N, 5) -> [x0, y0, x1, y1, class_id]
    filename: str
    image_id: int


def _load_coco() -> List[Sample]:
    """
    Load a small subset from the official YOLOX COCODataset for val2017 when available.
    If COCO data is missing, fall back to bundled sample images under ./assets with
    synthetic boxes so integration can still run.
    """
    samples: List[Sample] = []
    has_coco = COCO_ANN.exists() and COCO_IMG_DIR.exists()

    if has_coco:
        dataset = COCODataset(
            data_dir=str(COCO_ROOT),
            json_file="instances_val2017.json",
            name="val2017",
            img_size=(640, 640),
            preproc=ValTransform(legacy=False),
        )


    return dataset


@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    samples = _load_coco()
    if len(samples) == 0:
        raise RuntimeError("No samples found in COCO dataset.")
    return [
        PreprocessResponse(length=len(samples), data={"samples": samples}, state=DataStateType.training),
        PreprocessResponse(length=len(samples), data={"samples": samples}, state=DataStateType.validation),
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
def metadata_encoder(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
    sample = preprocess.data["samples"][idx]
    img, target, img_info, img_id = sample
    return {"orig_H": img_info[0], f"orig_W": img_info[1]}


def _to_bounding_boxes(sample: Sample) -> List[BoundingBox]:
    bboxes: List[BoundingBox] = []
    for x0, y0, x1, y1, cls in sample.boxes:
        class_id = int(cls)
        label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id)
        bboxes.append(
            BoundingBox(
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                label=label,
                confidence=1.0,
            )
        )
    return bboxes


@tensorleap_custom_visualizer("image_with_boxes", LeapDataType.ImageWithBBox)
def image_with_boxes_visualizer(image: np.ndarray, bboxes: np.ndarray) -> LeapImageWithBBox:
    # image is expected CHW float in [0,1]
    img = rescale_min_max(image.squeeze(0))
    boxes = []
    for x0, y0, x1, y1, cls in bboxes.squeeze(0):
        class_id = int(cls)
        label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id)
        boxes.append(
            BoundingBox(
                x=float(x0),
                y=float(y0),
                width=float(x1 - x0),
                height=float(y1 - y0),
                label=label,
                confidence=1.0,
            )
        )
    return LeapImageWithBBox(np.transpose(img, axes=[1,2,0]), boxes)


@tensorleap_custom_loss(name="bbox_l1_loss")
def bbox_l1_loss(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray) -> np.ndarray:
    """
    Simple L1 matching loss between predicted boxes (xyxy + scores/classes) and GT boxes.
    For each GT, find the closest predicted box by L1 distance over coordinates and average.
    """
    pred_bboxes = np.asarray(pred_bboxes)
    gt_bboxes = np.asarray(gt_bboxes)

    # Flatten any extra leading dims (e.g., batch or singleton) to align to (N, C)
    if pred_bboxes.ndim > 2:
        pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.shape[-1])
    if gt_bboxes.ndim > 2:
        gt_bboxes = gt_bboxes.reshape(-1, gt_bboxes.shape[-1])

    # Require at least 4 coords; otherwise return zero to keep decorator happy
    if pred_bboxes.size == 0 or gt_bboxes.size == 0 or pred_bboxes.shape[1] < 4 or gt_bboxes.shape[1] < 4:
        return np.array([0.0], dtype=np.float32)

    pred_xyxy = pred_bboxes[:, :4]
    gt_xyxy = gt_bboxes[:, :4]

    # pairwise L1 distances: [num_gt, num_pred]
    l1 = np.abs(gt_xyxy[:, None, :] - pred_xyxy[None, :, :]).sum(axis=2)
    best_pred_per_gt = l1.min(axis=1)

    # optional class mismatch penalty: add 1.0 if best class differs
    pred_cls = pred_bboxes[:, -1]
    gt_cls = gt_bboxes[:, -1]
    cls_diff = np.ones_like(best_pred_per_gt, dtype=np.float32)
    for i, gcls in enumerate(gt_cls.astype(int)):
        if l1.shape[1] > 0:
            j = l1[i].argmin()
            cls_diff[i] = 0.0 if (0 <= j < len(pred_cls) and int(pred_cls[j]) == gcls) else 1.0

    loss = (best_pred_per_gt + cls_diff).mean()
    return np.array([loss], dtype=np.float32)


def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in xyxy.
    a: [N,4], b: [M,4]
    returns [N,M]
    """
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / (union + 1e-8)


@tensorleap_custom_loss(name="yolox_total_loss")
def yolox_total_loss(pred_pre_nms: np.ndarray, gt_bboxes: np.ndarray) -> np.ndarray:
    """
    Compute YOLOX head loss by instantiating YOLOXHead and invoking get_losses.
    Uses the pre-NMS ONNX output (first output) assumed to be decoded xyxy + obj + class scores.
    """
    from yolox.models.yolo_head import YOLOXHead

    pred_pre_nms = np.asarray(pred_pre_nms)
    gt_bboxes = np.asarray(gt_bboxes)

    if pred_pre_nms.ndim == 3:
        pred_pre_nms = pred_pre_nms[0]
    if gt_bboxes.ndim == 3:
        gt_bboxes = gt_bboxes[0]

    if pred_pre_nms.size == 0 or gt_bboxes.size == 0 or pred_pre_nms.shape[1] < 6:
        zero = np.array([0.0], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
        {"total_loss": zero, "loss_iou": zero, "loss_obj": zero, "loss_cls": zero}

    # Limit anchors for speed
    K = min(200, pred_pre_nms.shape[0])
    preds_np = pred_pre_nms[:K]

    # Convert xyxy to xywh
    xyxy = preds_np[:, :4]
    cxcy = (xyxy[:, 0:2] + xyxy[:, 2:4]) / 2.0
    wh = xyxy[:, 2:4] - xyxy[:, 0:2]
    xywh = np.concatenate([cxcy, wh], axis=1)

    obj_score = preds_np[:, 4:5]
    cls_scores = preds_np[:, 5:]

    # Convert scores to logits (avoid inf)
    eps = 1e-6
    obj_logits = np.log(np.clip(obj_score, eps, 1 - eps) / np.clip(1 - obj_score, eps, 1 - eps))
    cls_logits = np.log(np.clip(cls_scores, eps, 1 - eps) / np.clip(1 - cls_scores, eps, 1 - eps))

    outputs = torch.from_numpy(np.concatenate([xywh, obj_logits, cls_logits], axis=1)).unsqueeze(0).float()

    # Build pseudo grid/stride so centers align with decoded boxes
    x_shifts = torch.from_numpy(cxcy[:, 0:1].T).float() - 0.5
    y_shifts = torch.from_numpy(cxcy[:, 1:2].T).float() - 0.5
    expanded_strides = torch.ones_like(x_shifts)

    # Labels: [batch, max_gt, 5] with class, cx, cy, w, h (xywh)
    gt_xyxy = gt_bboxes[:, :4]
    gt_cls = gt_bboxes[:, -1:]
    gt_cxcy = (gt_xyxy[:, 0:2] + gt_xyxy[:, 2:4]) / 2.0
    gt_wh = gt_xyxy[:, 2:4] - gt_xyxy[:, 0:2]
    labels_np = np.concatenate([gt_cls, gt_cxcy, gt_wh], axis=1).astype(np.float32)
    labels = torch.zeros((1, labels_np.shape[0], 5), dtype=torch.float32)
    labels[0, : labels_np.shape[0]] = torch.from_numpy(labels_np)

    # Instantiate head similar to yolox_s_ti_lite
    head = YOLOXHead(num_classes=len(COCO_CLASSES), width=0.50, in_channels=[256, 512, 1024], act="relu")

    loss, loss_iou, loss_obj, loss_cls, loss_l1, _ = head.get_losses(
        imgs=None,
        x_shifts=[x_shifts],
        y_shifts=[y_shifts],
        expanded_strides=[expanded_strides],
        labels=labels,
        outputs=outputs,
        origin_preds=None,
        dtype=outputs.dtype,
    )

    def _to_np(t):
        return t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1)

    return _to_np(loss)
    {
        "total_loss": _to_np(loss),
        "loss_iou": _to_np(loss_iou),
        "loss_obj": _to_np(loss_obj),
        "loss_cls": _to_np(loss_cls),
    }

def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes in xyxy.
    a: [N,4], b: [M,4]
    returns [N,M]
    """
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / (union + 1e-8)



# def yolox_total_loss(pred_pre_nms: np.ndarray, pred_post_nms: np.ndarray, gt_bboxes: np.ndarray) -> Dict[str, np.ndarray]:
#     """
#     Compute a YOLOX-style loss using the pre-NMS ONNX output (first output).
#     Returns total loss and individual components (IoU, objectness, classification).
#     """
#     pred_pre_nms = np.asarray(pred_pre_nms)
#     gt_bboxes = np.asarray(gt_bboxes)
#
#     # Flatten leading dims
#     if pred_pre_nms.ndim > 2:
#         pred_pre_nms = pred_pre_nms.reshape(-1, pred_pre_nms.shape[-1])
#     if gt_bboxes.ndim > 2:
#         gt_bboxes = gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
#
#     if pred_pre_nms.size == 0 or gt_bboxes.size == 0 or pred_pre_nms.shape[1] < 6:
#         zero = np.array([0.0], dtype=np.float32)
#         return {"total_loss": zero, "loss_iou": zero, "loss_obj": zero, "loss_cls": zero}
#
#     # Cap to keep computation light
#     K = min(100, pred_pre_nms.shape[0])
#     preds = torch.from_numpy(pred_pre_nms[:K]).float()
#     gts = torch.from_numpy(gt_bboxes).float()
#
#     pred_boxes = preds[:, :4]           # xyxy
#     obj_logits = preds[:, 4:5]          # objectness logit
#     cls_logits = preds[:, 5:]           # class logits
#
#     gt_boxes = gts[:, :4]
#     gt_classes = gts[:, -1].long()
#
#     # Match each GT to best pred via IoU
#     ious = _pairwise_iou_xyxy(gt_boxes, pred_boxes)
#     best_pred_idx = ious.argmax(dim=1)
#     best_pred_ious = ious.max(dim=1).values
#
#     # IoU loss (same formulation: 1 - iou^2)
#     loss_iou = (1 - best_pred_ious.clamp(min=0, max=1) ** 2).mean()
#
#     # Objectness targets: 1 for matched preds, 0 otherwise
#     obj_targets = torch.zeros_like(obj_logits)
#     obj_targets[best_pred_idx, 0] = 1.0
#     bce = nn.BCEWithLogitsLoss(reduction="mean")
#     loss_obj = bce(obj_logits, obj_targets)
#
#     # Classification targets one-hot on matched preds
#     if cls_logits.size(1) > 0:
#         cls_targets = torch.zeros_like(cls_logits)
#         for gi, pi in enumerate(best_pred_idx):
#             cls_id = int(gt_classes[gi].item())
#             if 0 <= cls_id < cls_logits.size(1):
#                 cls_targets[pi, cls_id] = 1.0
#         loss_cls = bce(cls_logits, cls_targets)
#     else:
#         loss_cls = torch.tensor(0.0, dtype=torch.float32)
#
#     reg_weight = 5.0
#     total = reg_weight * loss_iou + loss_obj + loss_cls
#
#     def _to_np(t):
#         return t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1)
#
#     return {
#         "total_loss": _to_np(total),
#         "loss_iou": _to_np(reg_weight * loss_iou),
#         "loss_obj": _to_np(loss_obj),
#         "loss_cls": _to_np(loss_cls),
#     }
