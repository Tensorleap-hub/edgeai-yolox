from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import yaml
import cv2
import numpy as np
import torch

from code_loader.utils import rescale_min_max
from code_loader.contract.datasetclasses import DataStateType, PreprocessResponse, SamplePreprocessResponse
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_metadata,
    tensorleap_custom_visualizer,
    tensorleap_custom_loss, tensorleap_custom_metric,
)
from yolox.utils.visualize import vis as yolox_vis
from yolox.data.datasets import COCO_CLASSES, COCODataset
from yolox.data.data_augment import ValTransform
from yolox.utils import bboxes_iou, postprocess
from yolox.models.losses import IOUloss


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


@dataclass
class Sample:
    image_path: Path
    boxes: np.ndarray  # shape (N, 5) -> [x0, y0, x1, y1, class_id]
    filename: str
    image_id: int


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
        img_size=(640, 640),
        preproc=ValTransform(legacy=False, visualize=True),
    )
    return dataset


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


@tensorleap_metadata("image info a")
def metadata_image_info_a(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    """
    Per-image stats for the COCO-person128 subset, including bbox counts and areas.
    Returns floats (or NaN) to satisfy possible_float_like_nan_types.
    """
    nan_default = float("nan")
    dataset = preprocess.data["samples"]
    img, target, img_info, img_id = dataset[idx]

    orig_h, orig_w = int(img_info[0]), int(img_info[1])
    resized_h, resized_w = dataset.annotations[idx][2]
    file_name = dataset.annotations[idx][3]

    num_boxes = int(target.shape[0])
    if num_boxes > 0:
        widths = target[:, 2] - target[:, 0]
        heights = target[:, 3] - target[:, 1]
        areas = widths * heights
        aspect = widths / (heights + 1e-9)
        classes, cls_counts = np.unique(target[:, 4], return_counts=True)
        mean_area = float(areas.mean())
        median_area = float(np.median(areas))
        max_area = float(areas.max())
        min_area = float(areas.min())
        mean_aspect = float(aspect.mean())
        num_unique_cls = float(len(classes))
    else:
        mean_area = median_area = max_area = min_area = mean_aspect = nan_default
        num_unique_cls = 0.0
        cls_counts = np.array([])

    return {
        "file_name": str(file_name),
        "image_id": float(img_id[0] if isinstance(img_id, np.ndarray) else img_id),
        "num_objects": float(num_boxes),
        "num_unique_classes": float(num_unique_cls),
        "orig_H": orig_h,
        "orig_W": orig_w,
        "resized_H": float(resized_h),
        "resized_W": float(resized_w),
        "mean_bbox_area": mean_area,
        "median_bbox_area": median_area,
        "max_bbox_area": max_area,
        "min_bbox_area": min_area,
        "mean_aspect_ratio": mean_aspect,
    }

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

@tensorleap_custom_visualizer("image", LeapDataType.Image)
def image_visualizer(image: np.ndarray, data: SamplePreprocessResponse,
) -> LeapImage:
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    # Convert model input back to displayable uint8 HWC and undo padding
    img_viz, r = post_process_image(image, meta_data)
    return LeapImage(img_viz, compress=False)

@tensorleap_custom_visualizer("image_with_boxes", LeapDataType.Image)
def image_with_boxes_visualizer(
    image: np.ndarray,
    bboxes: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImage:
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    # Convert model input back to displayable uint8 HWC and undo padding
    img_viz, r = post_process_image(image, meta_data)

    bboxes = bboxes.copy().squeeze(0)
    boxes_arr = bboxes[:,:4]
    cls_ids = bboxes[:,-1]
    scores = np.ones(len(boxes_arr), dtype=np.float32)
    # denormalize
    boxes_arr[:, [0, 2]] /= r
    boxes_arr[:, [1, 3]] /= r

    img_viz = yolox_vis(img_viz.copy(), boxes_arr, scores, cls_ids, conf=0.0, class_names=CLASSES)

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
    if preds.shape[1]==8400:
        boxes = postprocess(torch.tensor(preds), conf_thre=0.3, nms_thre=0.45,
                        num_classes=NUM_CLASSES, class_agnostic=True)[0]
    else:
        boxes = preds.copy()[0,::]
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    img_viz, r = post_process_image(image, meta_data)
    if boxes is None or boxes.size == 0:
        return LeapImage(img_viz, compress=False)
    boxes = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    boxes_arr = boxes[:,:4]
    cls_ids =boxes[:,-1]
    scores_obj = boxes[:,4]
    # denormalize
    boxes_arr[:, [0, 2]] /= r
    boxes_arr[:, [1, 3]] /= r
    cls_ids_np = cls_ids.astype(np.int32)
    img_viz = yolox_vis(img_viz.copy(), boxes_arr, scores_obj, cls_ids_np, conf=0.0, class_names=CLASSES)

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

    outputs_cat = torch.cat(outputs_decoded, dim=1)


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

    return loss.detach().cpu().numpy().astype(np.float32), {'loss_iou':loss_iou.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_obj':loss_obj.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_cls':loss_cls.unsqueeze(0).cpu().numpy().astype(np.float32),
                                                            'loss_l1':np.array([loss_l1]).astype(np.float32),}

@tensorleap_custom_loss(name="total_loss")
def total_loss(pred80, pred40, pred20, gt_bboxes: np.ndarray):
    total_loss, _ = yolox_head_loss_raw(pred80.copy(), pred40.copy(), pred20.copy(), gt_bboxes.copy())
    return total_loss

@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80, pred40, pred20, gt_bboxes: np.ndarray) -> np.ndarray:
    _, parts = yolox_head_loss_raw(pred80.copy(), pred40.copy(), pred20.copy(), gt_bboxes.copy())
    return parts

# --------------------------------------------------------------------------- #
# Detection metrics (precision / recall / F1 / accuracy) using VOC-style IoU  #
# --------------------------------------------------------------------------- #

def _match_detections(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5):
    """Greedy one-to-one matching of predicted boxes to GT by IoU."""
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return [], set(), set(range(len(gt_boxes)))

    ious = bboxes_iou(torch.from_numpy(pred_boxes[:, :4]), torch.from_numpy(gt_boxes[:, :4])).numpy()
    matches = []
    used_gt = set()
    used_pred = set()
    for p_idx in np.argsort(-ious.max(axis=1)):
        if p_idx in used_pred:
            continue
        gt_idx = int(np.argmax(ious[p_idx]))
        if gt_idx in used_gt:
            continue
        if ious[p_idx, gt_idx] >= iou_thresh and pred_boxes[p_idx, 4] == gt_boxes[gt_idx, 4]:
            matches.append((p_idx, gt_idx))
            used_gt.add(gt_idx)
            used_pred.add(p_idx)
    unused_pred = set(range(len(pred_boxes))) - used_pred
    unused_gt = set(range(len(gt_boxes))) - used_gt
    return matches, unused_pred, unused_gt


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
    if preds.ndim > 2:
        decoded = postprocess(torch.tensor(preds), conf_thre=0.3, nms_thre=0.45,
                              num_classes=NUM_CLASSES, class_agnostic=True)[0]
        if decoded is None:
            decoded = np.zeros((0, 7), dtype=np.float32)
        else:
            decoded = decoded.cpu().numpy()
    else:
        decoded = preds

    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    pred_boxes = decoded[:, :5] if decoded.size else np.zeros((0, 5), dtype=np.float32)
    pred_cls = decoded[:, -1:] if decoded.size else np.zeros((0, 1), dtype=np.float32)
    pred_boxes = np.concatenate([pred_boxes[:, :4], pred_cls], axis=1) if pred_boxes.size else pred_boxes

    matches, unused_pred, unused_gt = _match_detections(pred_boxes, gt, iou_thresh=0.5)
    tp = float(len(matches))
    fp = float(len(unused_pred))
    fn = float(len(unused_gt))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = tp / (tp + fp + fn + 1e-9)

    return {
        "F1": np.array([f1], dtype=np.float32),
        "recall": np.array([recall], dtype=np.float32),
        "precision": np.array([precision], dtype=np.float32),
        "accuracy": np.array([accuracy], dtype=np.float32),
    }
