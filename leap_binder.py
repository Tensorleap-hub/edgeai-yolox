from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml
import cv2
import numpy as np
import torch
from code_loader.contract.responsedataclasses import BoundingBox

from code_loader.utils import rescale_min_max
from code_loader.contract.datasetclasses import DataStateType, PreprocessResponse, SamplePreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage, LeapImageWithBBox
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_metadata,
    tensorleap_custom_visualizer,
    tensorleap_custom_loss, tensorleap_custom_metric,
)

from tools.image_tools import estimate_noise
from tools.bbox_utils import filter_valid_gt, bbox_stats, xyxy_to_bounding_boxes
from tools.metrics_utils import prf1_from_counts, match_detections, match_detections_iou_only
from tools.pred_decode import decode_preds
from tools.fisheye import fisheye_stats
from yolox.data.datasets import COCO_CLASSES, COCODataset
from yolox.data.data_augment import ValTransform
from yolox.utils import bboxes_iou


CONFIG_PATH = Path(__file__).with_name("tensorleap_config.yaml")
def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}  # dict from YAML (or empty dict)

cfg = load_config()
CLASSES = cfg["CLASSES"]
DATA_ROOT = Path(cfg["DATA_ROOT"])
NUM_CLASSES = len(CLASSES)
ANN_ROOT = DATA_ROOT / "annotations" / cfg["VAL_JSON"]
STRIDES = cfg["STRIDES"]
LIMIT_SAMPLES = cfg["LIMIT_SAMPLES"]
OBJ_THRESH = float(cfg.get("OBJ_THRESH", 0.3))
IOU_THRESH = float(cfg.get("IOU_THRESH", 0.5))
FOCAL_LENGTH = float(cfg.get("FOCAL_LENGTH", 300.0))
IMG_SIZE = tuple(cfg.get("IMG_SIZE", [640, 640]))
CONF_THRESH = float(cfg.get("CONF_THRESH", 0.3))
NMS_THRESH = float(cfg.get("NMS_THRESH", 0.45))
DET_IOU_THRESH = float(cfg.get("DET_IOU_THRESH", 0.5))

# ============================================================================
#                           DATA STRUCTURES
# ============================================================================


@dataclass
class Sample:
    image_path: Path
    boxes: np.ndarray  # shape (N, 5) -> [x0, y0, x1, y1, class_id]
    filename: str
    image_id: int


# ============================================================================
#                           DATASET LOADING
# ============================================================================

def _load_coco(name="val") -> COCODataset:
    """Load coco-person128 split; fail loudly if missing."""
    has_coco = ANN_ROOT.exists()
    if not has_coco:
        raise FileNotFoundError(f"Dataset not found in {ANN_ROOT}")

    if name == "val":
        ds_name = cfg["VAL_NAME"]
        json_file = cfg["VAL_JSON"]
    elif name == "unlabeled":
        ds_name = cfg["UNLABELED_NAME"]
        json_file = cfg["UNLABELED_JSON"]
        if ds_name is None or json_file is None:
            return None
    elif name == "train":
        ds_name = cfg["TRAIN_NAME"]
        json_file = cfg["TRAIN_JSON"]
    else:
        raise KeyError("Unknown dataset")

    dataset = COCODataset(
        data_dir=str(DATA_ROOT),
        json_file=json_file,
        name=ds_name,
        img_size=IMG_SIZE,
        preproc=ValTransform(legacy=False, visualize=True),
    )
    return dataset


# ============================================================================
#                              PREPROCESS
# ============================================================================

@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    train_data = _load_coco('train')
    val_data = _load_coco('val')
    unlabeled_data = _load_coco('unlabeled')

    if not train_data or not val_data:
        raise RuntimeError("No samples found in COCO dataset.")

    datasets = [
        (train_data, DataStateType.training),
        (val_data, DataStateType.validation),
    ]

    if unlabeled_data is not None:
        datasets.append((unlabeled_data, DataStateType.unlabeled))

    responses = []
    for data, state in datasets:
        length = len(data)
        if LIMIT_SAMPLES is not None:
            length = min(length, LIMIT_SAMPLES)

        responses.append(
            PreprocessResponse(
                length=length,
                data={"samples": data},
                state=state
            )
        )

    return responses



# ============================================================================
#                       INPUT / GT ENCODERS
# ============================================================================

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


# ============================================================================
#                           METADATA
# ============================================================================

def pred_to_class(idx:int) -> str:
    return CLASSES[idx]

def count_classes(cls_ids:list) -> Dict[str, int]:
    counts = {cls: 0 for cls in CLASSES}
    for ids in cls_ids:
        cls = pred_to_class(ids)
        counts[cls] += 1
    return counts


def _to_numpy(x: Any) -> np.ndarray:
    """Accept np.ndarray or torch.Tensor and return np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@tensorleap_metadata("image info a")
def metadata_image_info_a(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    nan_default = float("nan")
    dataset = preprocess.data["samples"]
    img, target, img_info, img_id = dataset[idx]

    orig_h, orig_w = int(img_info[0]), int(img_info[1])
    resized_h, resized_w = dataset.annotations[idx][2]
    file_name = dataset.annotations[idx][3]
    r = min(resized_h / orig_h, resized_w / orig_w) if orig_h > 0 and orig_w > 0 else 1.0

    img_np = np.transpose(img, axes=(1, 2, 0)).astype("uint8")
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_est = estimate_noise(image=img,method='laplacian')
    target_np = _to_numpy(target)
    stats, _, _ = bbox_stats(target_np, cls_col=4, nan_default=nan_default)
    valid = (target_np[:, 2] > target_np[:, 0]) & (target_np[:, 3] > target_np[:, 1]) if target_np.size else []
    target_xyxy = target_np[valid][:, :4] if target_np.size else np.zeros((0, 4), dtype=np.float32)
    target_xyxy = target_xyxy / r if target_xyxy.size and r > 0 else target_xyxy
    fisheye_meta = fisheye_stats(
        target_xyxy,
        orig_w=orig_w,
        orig_h=orig_h,
        focal_length=FOCAL_LENGTH,
        nan_default=nan_default,
    )
    # Class counts (uses your existing helpers/constants)
    cls_ids = target[:, -1]  # keep your convention
    counts = count_classes(list(cls_ids.astype(int))) if int(stats["num_objects"]) > 0 else {cls: 0 for cls in CLASSES}
    return {
        "file_name": str(file_name),
        "image_id": float(img_id[0] if isinstance(img_id, np.ndarray) else img_id),
        "orig_H": float(orig_h),
        "orig_W": float(orig_w),
        "resized_H": float(resized_h),
        "resized_W": float(resized_w),
        "sharpness": float(sharpness),
        "noise_est": float(noise_est),
        **stats,
        **fisheye_meta,
        **counts,
    }


# ============================================================================
#                       PREDICTION METADATA METRICS
# ============================================================================

@tensorleap_custom_metric("prediction_metadata",  compute_insights={'num_objects': False,
                                                                    'num_unique_classes': False,
                                                                    'mean_bbox_area': False,
                                                                    'median_bbox_area': False,
                                                                    'max_bbox_area': False,
                                                                    'min_bbox_area': False,
                                                                    'mean_aspect_ratio': False,
                                                                    'mean_conf': False,
                                                                    'median_conf': False,
                                                                    'person': False,
                                                                    'light_vehicle': False,
                                                                    'machine': False}
)
def pred_statistics(prediction: np.ndarray, image: np.ndarray, data: SamplePreprocessResponse) -> Dict[str, Any]:

    nan_default = float("nan")

    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    _, r = post_process_image(image, meta_data)

    boxes = prediction.copy()[0,::]

    # denormalize xyxy
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r

    stats, _, _ = bbox_stats(boxes, cls_col=4, nan_default=nan_default)

    # Class counts (uses your existing helpers/constants)
    cls_ids = boxes[:, -1]  # keep your convention
    counts = count_classes(list(cls_ids.astype(int))) if int(stats["num_objects"]) > 0 else {cls: 0 for cls in CLASSES}

    # Confidence stats (keep your convention)
    if int(stats["num_objects"]) > 0:
        conf = boxes[:, -2]
        conf_stats = {
            "mean_conf": float(conf.mean()),
            "median_conf": float(np.median(conf)),
            "max_conf": float(conf.max()),
            "min_conf": float(conf.min()),
        }
    else:
        conf_stats = {
            "mean_conf": nan_default,
            "median_conf": nan_default,
            "max_conf": nan_default,
            "min_conf": nan_default,
        }

    pred_stats = {
        **stats,
        **conf_stats,
        **counts,
    }
    pred_stats = {key:np.array([value]) for key, value in pred_stats.items()}
    return pred_stats


# ============================================================================
#                           IMAGE POST-PROCESSING
# ============================================================================

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
    img_viz = cv2.resize(img_viz, (int(orig_W), int(orig_H)), interpolation=cv2.INTER_LINEAR)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
    return img_viz, r


# ============================================================================
#                               VISUALIZERS
# ============================================================================

@tensorleap_custom_visualizer("image", LeapDataType.Image)
def image_visualizer(image: np.ndarray, data: SamplePreprocessResponse,
) -> LeapImage:
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    # Convert model input back to displayable uint8 HWC and undo padding
    img_viz, r = post_process_image(image, meta_data)
    return LeapImage(img_viz, compress=False)


@tensorleap_custom_visualizer("image_with_pred_boxes", LeapDataType.ImageWithBBox)
def image_with_pred_boxes_visualizer(
    image: np.ndarray,
    preds: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImageWithBBox:
    """
    Visualize predictions in (xyxy + obj + class scores) format from pre-NMS output.
    """
    boxes = decode_preds(
        preds,
        conf_thre=CONF_THRESH,
        nms_thre=NMS_THRESH,
        num_classes=NUM_CLASSES,
        class_agnostic=True,
    )
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    img_viz, r = post_process_image(image, meta_data)
    if boxes is None or boxes.size == 0:
        return LeapImageWithBBox(img_viz, [])

    boxes_arr = boxes[:, :4].astype(np.float32)
    cls_ids = boxes[:, -1].astype(int)
    scores_obj = boxes[:, 4] if boxes.shape[1] > 4 else np.ones(len(boxes_arr), dtype=np.float32)
    leap_boxes = xyxy_to_bounding_boxes(
        boxes_arr,
        cls_ids=cls_ids,
        scores=scores_obj,
        r=r,
        image_shape=img_viz.shape[:2],
        labels=CLASSES,
    )

    return LeapImageWithBBox(img_viz, leap_boxes)


@tensorleap_custom_visualizer("image_with_gt_and_pred_boxes", LeapDataType.ImageWithBBox)
def image_with_gt_and_pred_boxes_visualizer(
    image: np.ndarray,
    bboxes: np.ndarray,
    preds: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImageWithBBox:
    """
    Visualize GT and predictions together using BoundingBox metadata.
    GT boxes are tagged with metadata {"source": "gt"} and predictions with {"source": "pred"}.
    """
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    img_viz, r = post_process_image(image, meta_data)

    leap_boxes = []

    # ---- GT boxes (xyxy in resized pixels) ----
    gt = bboxes.copy().squeeze(0)
    if gt.size > 0:
        valid = (gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1])
        gt = gt[valid]
        if gt.size > 0:
            gt_xyxy = gt[:, :4].astype(np.float32)
            gt_cls_ids = gt[:, -1].astype(int)
            gt_scores = np.ones(len(gt_xyxy), dtype=np.float32)
            leap_boxes.extend(
                xyxy_to_bounding_boxes(
                    gt_xyxy,
                    cls_ids=gt_cls_ids,
                    scores=gt_scores,
                    r=r,
                    image_shape=img_viz.shape[:2],
                    labels=CLASSES,
                    label_suffix="_gt",
                    metadata={"source": "gt"},
                )
            )

    # ---- Pred boxes ----
    if preds is not None:
        pred_boxes = decode_preds(
            preds,
            conf_thre=CONF_THRESH,
            nms_thre=NMS_THRESH,
            num_classes=NUM_CLASSES,
            class_agnostic=True,
        )

        if pred_boxes is not None and pred_boxes.size != 0:
            pred_xyxy = pred_boxes[:, :4].astype(np.float32)
            pred_cls_ids = pred_boxes[:, -1].astype(int)
            pred_scores = pred_boxes[:, 4] if pred_boxes.shape[1] > 4 else np.ones(
                len(pred_xyxy), dtype=np.float32
            )
            leap_boxes.extend(
                xyxy_to_bounding_boxes(
                    pred_xyxy,
                    cls_ids=pred_cls_ids,
                    scores=pred_scores,
                    r=r,
                    image_shape=img_viz.shape[:2],
                    labels=CLASSES,
                    label_suffix="_pred",
                    metadata={"source": "pred"},
                )
            )

    return LeapImageWithBBox(img_viz, leap_boxes)


# ============================================================================
#                               LOSSES
# ============================================================================


# --------------------------------------------------------------------------- #
# YOLOX head loss using raw head outputs (pre-decode)                         #
# --------------------------------------------------------------------------- #

def yolox_head_loss_raw(pred80, pred40, pred20, gt_bboxes: np.ndarray):
    """
    Compute YOLOX loss from raw head outputs (per-level tensors before decode/NMS).

    Expected raw head outputs:
      - pred80 / pred40 / pred20: BCHW tensors where
        C = 4 (reg) + 1 (obj) + num_classes (cls)
      - B must be 1 (this function is single-sample only)
      - Layout must be BCHW (NCHW); NHWC is not supported

    Ground truth:
      - gt_bboxes: [..., 5] with [x1, y1, x2, y2, class] in resized image pixels
      - If batched (3D), only the first batch element is used

    Returns:
      - total loss as a scalar numpy array [[float32]]
      - a dict of loss parts: loss_iou, loss_obj, loss_cls, loss_l1
    """

    from yolox.models.yolo_head import YOLOXHead
    # Guard against unsupported batch sizes or layout at the entry point.
    head0 = np.asarray(pred80)
    if head0.ndim != 4:
        raise ValueError(
            f"yolox_head_loss_raw expects BCHW head outputs; got shape {head0.shape}"
        )
    if head0.shape[0] != 1:
        raise ValueError(
            f"yolox_head_loss_raw only supports batch size 1; got B={head0.shape[0]}"
        )
    head_outs = [pred80.copy(), pred40.copy(), pred20.copy()]
    # Normalize inputs
    if isinstance(head_outs, (list, tuple)):
        outs_np = [np.asarray(o) for o in head_outs]
    else:
        # ONNXRuntime may deliver a flat tuple; handle single-level too
        outs_np = [np.asarray(head_outs)]
    strides = np.asarray(STRIDES).flatten()
    gt_bboxes = np.asarray(gt_bboxes).copy()

    if gt_bboxes.ndim == 3:
        gt_bboxes = gt_bboxes[0]
    # if len(outs_np) == 0 or gt_bboxes.size == 0:
    #     return np.array([0.0], dtype=np.float32)

    num_levels = len(outs_np)
    num_classes = outs_np[0].shape[1] - 5  # C = 4+1+num_classes

    # Build a lightweight head just to reuse get_output_and_grid / get_losses.
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
        # Decode to absolute bbox predictions (cx, cy, w, h) in resized pixels.
        output, grid = head.get_output_and_grid(out_t, k, stride, out_t.type())
        outputs_decoded.append(output)
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(torch.full((1, grid.shape[1]), float(stride), dtype=dtype))

    outputs_cat = torch.cat(outputs_decoded, dim=1)


    # Build labels tensor [B, max_gt, 5] with class-first and cxcywh.
    gt_xyxy = torch.from_numpy(gt_bboxes[:, :4]).to(dtype)
    gt_cls = torch.from_numpy(gt_bboxes[:, -1:]).to(dtype)
    gt_cxcy = (gt_xyxy[:, 0:2] + gt_xyxy[:, 2:4]) / 2.0
    gt_wh = (gt_xyxy[:, 2:4] - gt_xyxy[:, 0:2]).clamp(min=1e-6)
    labels = torch.zeros((1, gt_xyxy.shape[0], 5), dtype=dtype)
    labels[0, :, 0:1] = gt_cls
    labels[0, :, 1:3] = gt_cxcy
    labels[0, :, 3:5] = gt_wh

    # Compute losses using YOLOX head assignment + loss definitions.
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

    return loss.detach().cpu().numpy().astype(np.float32), {'loss_iou':loss_iou.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_obj':loss_obj.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_cls':loss_cls.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_l1':np.array([loss_l1]).astype(np.float32),}

@tensorleap_custom_loss(name="total_loss")
def total_loss(pred80, pred40, pred20, gt_bboxes: np.ndarray):
    total_loss, _ = yolox_head_loss_raw(pred80.copy(), pred40.copy(), pred20.copy(), gt_bboxes.copy())
    return total_loss

# ============================================================================
# METRICS
# ============================================================================

@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80, pred40, pred20, gt_bboxes: np.ndarray) -> np.ndarray:
    _, parts = yolox_head_loss_raw(pred80.copy(), pred40.copy(), pred20.copy(), gt_bboxes.copy())
    return parts

# ============================================================================
# DETECTION METRICS
# ============================================================================

@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def ious(preds: np.ndarray, gt_bboxes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Greedy one-to-one IoU matching.
    Returns per-class mean IoU (over GT instances) and mean sample IoU.
    """
    default_value = np.ones(1, dtype=np.float32) * float("nan")
    iou_dic = {cls_name: default_value for cls_name in CLASSES}

    # Decode predictions to xyxy (+ obj + cls) format.
    decoded = decode_preds(
        preds,
        conf_thre=CONF_THRESH,
        nms_thre=NMS_THRESH,
        num_classes=NUM_CLASSES,
        class_agnostic=True,
    )

    # Flatten GT to [N, 5] and drop invalid rows.
    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    gt = filter_valid_gt(gt)

    pred_xyxy = decoded[:, :4] if decoded.size else np.zeros((0, 4), dtype=np.float32)
    gt_xyxy = gt[:, :4] if gt.size else np.zeros((0, 4), dtype=np.float32)

    n_gt = gt_xyxy.shape[0]
    n_pred = pred_xyxy.shape[0]

    if n_gt == 0 and n_pred == 0:
        iou_dic["mean sample iou"] = default_value
        return iou_dic

    # IoU matrix in resized-pixel space.
    if n_gt > 0 and n_pred > 0:
        iou_mat = bboxes_iou(
            torch.from_numpy(gt_xyxy),
            torch.from_numpy(pred_xyxy),
        ).numpy()
    else:
        iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)

    # Greedy one-to-one matching over predictions.
    used_gt = np.zeros(n_gt, dtype=bool)
    assigned_iou_per_gt = np.zeros(n_gt, dtype=np.float32)
    iou_per_pred = np.zeros(n_pred, dtype=np.float32)

    for j in range(n_pred):
        if n_gt == 0:
            break
        i = int(np.argmax(iou_mat[:, j]))
        best = float(iou_mat[i, j])
        if not used_gt[i]:
            iou_per_pred[j] = best
            assigned_iou_per_gt[i] = best
            used_gt[i] = True

    # Mean over all instances (preds + unmatched GTs as zero).
    all_instance_ious = np.concatenate([iou_per_pred, np.zeros(np.sum(~used_gt), dtype=np.float32)])
    mean_iou_sample = np.expand_dims(all_instance_ious.mean(), axis=0).astype(np.float32)

    # Per-class mean IoU over GT assignments.
    if n_gt > 0:
        cls_gt = gt[:, 4].astype(int)
        for cls_id, cls_name in enumerate(CLASSES):
            mask_c = cls_gt == cls_id
            if mask_c.any():
                iou_dic[cls_name] = np.expand_dims(assigned_iou_per_gt[mask_c].mean(), axis=0).astype(np.float32)

    iou_dic["mean sample iou"] = mean_iou_sample
    return iou_dic


@tensorleap_custom_metric("objectness_prf1", direction=MetricDirection.Upward)
def objectness_prf1(preds: np.ndarray, gt_bboxes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Class-agnostic objectness PRF1 using obj threshold and IoU-only matching.
    """
    decoded = decode_preds(
        preds,
        conf_thre=0.0,
        nms_thre=NMS_THRESH,
        num_classes=NUM_CLASSES,
        class_agnostic=True,
    )

    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    gt = filter_valid_gt(gt)

    pred_xyxy = decoded[:, :4] if decoded.size else np.zeros((0, 4), dtype=np.float32)
    pred_obj = decoded[:, 4] if decoded.size else np.zeros((0,), dtype=np.float32)
    keep = pred_obj >= OBJ_THRESH
    pred_xyxy = pred_xyxy[keep]

    gt_xyxy = gt[:, :4] if gt.size else np.zeros((0, 4), dtype=np.float32)

    matches, unused_pred, unused_gt = match_detections_iou_only(
        pred_xyxy, gt_xyxy, iou_thresh=IOU_THRESH
    )
    tp = float(len(matches))
    fp = float(len(unused_pred))
    fn = float(len(unused_gt))
    precision, recall, f1, accuracy = prf1_from_counts(tp, fp, fn)

    return {
        "F1": np.array([f1], dtype=np.float32),
        "recall": np.array([recall], dtype=np.float32),
        "precision": np.array([precision], dtype=np.float32),
        "accuracy": np.array([accuracy], dtype=np.float32),
    }


@tensorleap_custom_metric("detection_prf1", direction=MetricDirection.Upward)
def detection_prf1(preds: np.ndarray, gt_bboxes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute precision/recall/F1/accuracy for detections on a single sample.

    Args:
        preds: model detections or raw head outputs. If raw heads (ndim>2), they
               are decoded with default thresholds; otherwise expected shape
               [N, 7] with [x1, y1, x2, y2, obj, cls_conf, cls].
        gt_bboxes: ground-truth boxes [N,5] in xyxy + class format.
    """
    decoded = decode_preds(
        preds,
        conf_thre=CONF_THRESH,
        nms_thre=NMS_THRESH,
        num_classes=NUM_CLASSES,
        class_agnostic=True,
    )

    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    gt = filter_valid_gt(gt)
    pred_boxes = decoded[:, :5] if decoded.size else np.zeros((0, 5), dtype=np.float32)
    pred_cls = decoded[:, -1:] if decoded.size else np.zeros((0, 1), dtype=np.float32)
    pred_boxes = np.concatenate([pred_boxes[:, :4], pred_cls], axis=1) if pred_boxes.size else pred_boxes

    matches, unused_pred, unused_gt = match_detections(pred_boxes, gt, iou_thresh=DET_IOU_THRESH)
    tp = float(len(matches))
    fp = float(len(unused_pred))
    fn = float(len(unused_gt))

    precision, recall, f1, accuracy = prf1_from_counts(tp, fp, fn)

    metrics = {
        "F1": np.array([f1], dtype=np.float32),
        "recall": np.array([recall], dtype=np.float32),
        "precision": np.array([precision], dtype=np.float32),
        "accuracy": np.array([accuracy], dtype=np.float32),
    }

    # Per-class metrics.
    pred_cls = pred_boxes[:, 4].astype(int) if pred_boxes.size else np.zeros((0,), dtype=int)
    gt_cls = gt[:, 4].astype(int) if gt.size else np.zeros((0,), dtype=int)
    matches = np.asarray(matches, dtype=int) if len(matches) else np.zeros((0, 2), dtype=int)
    unused_pred = np.asarray(sorted(unused_pred), dtype=int) if len(unused_pred) else np.zeros((0,), dtype=int)
    unused_gt = np.asarray(sorted(unused_gt), dtype=int) if len(unused_gt) else np.zeros((0,), dtype=int)

    for cls_id, cls_name in enumerate(CLASSES):
        tp_c = 0
        if matches.size:
            tp_c = int((gt_cls[matches[:, 1]] == cls_id).sum())
        fp_c = int((pred_cls[unused_pred] == cls_id).sum()) if unused_pred.size else 0
        fn_c = int((gt_cls[unused_gt] == cls_id).sum()) if unused_gt.size else 0

        precision_c = tp_c / (tp_c + fp_c + 1e-9) if (tp_c + fp_c) > 0 else float("nan")
        recall_c = tp_c / (tp_c + fn_c + 1e-9) if (tp_c + fn_c) > 0 else float("nan")
        if np.isnan(precision_c) or np.isnan(recall_c) or (precision_c + recall_c) == 0:
            f1_c = float("nan")
        else:
            f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
        accuracy_c = (
            tp_c / (tp_c + fp_c + fn_c + 1e-9) if (tp_c + fp_c + fn_c) > 0 else float("nan")
        )

        # metrics[f"precision_{cls_name}"] = np.array([precision_c], dtype=np.float32)
        # metrics[f"recall_{cls_name}"] = np.array([recall_c], dtype=np.float32)
        metrics[f"F1_{cls_name}"] = np.array([f1_c], dtype=np.float32)
        # metrics[f"accuracy_{cls_name}"] = np.array([accuracy_c], dtype=np.float32)

    return metrics
