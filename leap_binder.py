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
from yolox.data.datasets import COCO_CLASSES, COCODataset
from yolox.data.data_augment import ValTransform
from yolox.utils import bboxes_iou, postprocess


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
EPS = 1e-9
OBJ_THRESH = float(cfg.get("OBJ_THRESH", 0.3))
IOU_THRESH = float(cfg.get("IOU_THRESH", 0.5))
FOCAL_LENGTH = float(cfg.get("FOCAL_LENGTH", 300.0))
IMG_SIZE = tuple(cfg.get("IMG_SIZE", [640, 640]))
CONF_THRESH = float(cfg.get("CONF_THRESH", 0.3))
NMS_THRESH = float(cfg.get("NMS_THRESH", 0.45))
DET_IOU_THRESH = float(cfg.get("DET_IOU_THRESH", 0.5))


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
        img_size=IMG_SIZE,
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


def _bbox_stats(
    boxes: np.ndarray,
    *,
    cls_col: int = 4,
    nan_default: float = float("nan"),
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute bbox area/aspect and unique-class stats.
    Returns:
      stats: dict of scalar floats
      classes: unique class ids (np.ndarray)
      cls_counts: counts per unique class (np.ndarray)
    Expected box format: [x1, y1, x2, y2, cls, ...] (cls_col points to class id).
    """
    num_boxes = int(boxes.shape[0])
    if num_boxes == 0:
        return (
            {
                "num_objects": 0.0,
                "num_unique_classes": 0.0,
                "mean_bbox_area": nan_default,
                "median_bbox_area": nan_default,
                "max_bbox_area": nan_default,
                "min_bbox_area": nan_default,
                "mean_aspect_ratio": nan_default,
            },
            np.array([]),
            np.array([]),
        )

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    aspect = widths / (heights + EPS)

    classes, cls_counts = np.unique(boxes[:, cls_col], return_counts=True)

    return (
        {
            "num_objects": float(num_boxes),
            "num_unique_classes": float(len(classes)),
            "mean_bbox_area": float(areas.mean()),
            "median_bbox_area": float(np.median(areas)),
            "max_bbox_area": float(areas.max()),
            "min_bbox_area": float(areas.min()),
            "mean_aspect_ratio": float(aspect.mean()),
        },
        classes,
        cls_counts,
    )


def get_fisheye_metadata(
    bb_xywh: np.ndarray,
    image_width: int,
    image_height: int,
    focal_length: float,
) -> Dict[str, float]:
    """Compute fisheye-related metadata for a single bbox in original pixel space."""
    # Optical center
    cx, cy = image_width / 2.0, image_height / 2.0

    # Bounding box center (u, v)
    u = float(bb_xywh[0] + bb_xywh[2] / 2.0)
    v = float(bb_xywh[1] + bb_xywh[3] / 2.0)

    # Radial distance normalized to [0, 1]
    max_r = float(np.sqrt(cx**2 + cy**2))
    r = float(np.sqrt((u - cx) ** 2 + (v - cy) ** 2))
    radial_dist = r / max_r if max_r > 0 else 0.0

    # Angular position
    phi = float(np.arctan2(v - cy, u - cx))

    # Incident angle (equidistant model: r = f * theta)
    theta = float(r / focal_length) if focal_length > 0 else 0.0

    # Distortion factor (simple heuristic)
    distortion_factor = float(np.sin(theta)) if theta < (np.pi / 2.0) else 1.0

    return {
        "radial_dist": radial_dist,
        "phi": phi,
        "theta": theta,
        "distortion_factor": distortion_factor,
    }


def _fisheye_stats(
    boxes_xyxy: np.ndarray,
    *,
    orig_w: int,
    orig_h: int,
    focal_length: float,
    nan_default: float,
) -> Dict[str, float]:
    """Aggregate fisheye metadata across all bboxes in a sample."""
    if boxes_xyxy.size == 0:
        return {
            "fisheye_radial_dist_mean": nan_default,
            "fisheye_radial_dist_median": nan_default,
            "fisheye_radial_dist_min": nan_default,
            "fisheye_radial_dist_max": nan_default,
            "fisheye_phi_mean": nan_default,
            "fisheye_phi_median": nan_default,
            "fisheye_phi_min": nan_default,
            "fisheye_phi_max": nan_default,
            "fisheye_theta_mean": nan_default,
            "fisheye_theta_median": nan_default,
            "fisheye_theta_min": nan_default,
            "fisheye_theta_max": nan_default,
            "fisheye_distortion_mean": nan_default,
            "fisheye_distortion_median": nan_default,
            "fisheye_distortion_min": nan_default,
            "fisheye_distortion_max": nan_default,
        }

    xywh = boxes_xyxy.copy()
    xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
    xywh[:, 3] = xywh[:, 3] - xywh[:, 1]

    radial = []
    phi = []
    theta = []
    distortion = []
    for bb in xywh:
        meta = get_fisheye_metadata(bb, orig_w, orig_h, focal_length)
        radial.append(meta["radial_dist"])
        phi.append(meta["phi"])
        theta.append(meta["theta"])
        distortion.append(meta["distortion_factor"])

    radial = np.asarray(radial, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    distortion = np.asarray(distortion, dtype=np.float32)

    return {
        "fisheye_radial_dist_mean": float(radial.mean()),
        "fisheye_radial_dist_median": float(np.median(radial)),
        "fisheye_radial_dist_min": float(radial.min()),
        "fisheye_radial_dist_max": float(radial.max()),
        "fisheye_phi_mean": float(phi.mean()),
        "fisheye_phi_median": float(np.median(phi)),
        "fisheye_phi_min": float(phi.min()),
        "fisheye_phi_max": float(phi.max()),
        "fisheye_theta_mean": float(theta.mean()),
        "fisheye_theta_median": float(np.median(theta)),
        "fisheye_theta_min": float(theta.min()),
        "fisheye_theta_max": float(theta.max()),
        "fisheye_distortion_mean": float(distortion.mean()),
        "fisheye_distortion_median": float(np.median(distortion)),
        "fisheye_distortion_min": float(distortion.min()),
        "fisheye_distortion_max": float(distortion.max()),
    }


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
    stats, _, _ = _bbox_stats(target_np, cls_col=4, nan_default=nan_default)
    valid = (target_np[:, 2] > target_np[:, 0]) & (target_np[:, 3] > target_np[:, 1]) if target_np.size else []
    target_xyxy = target_np[valid][:, :4] if target_np.size else np.zeros((0, 4), dtype=np.float32)
    target_xyxy = target_xyxy / r if target_xyxy.size and r > 0 else target_xyxy
    fisheye_stats = _fisheye_stats(
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
        **fisheye_stats,
        **counts,
    }


@tensorleap_custom_metric("prediction_metadata",  compute_insights={'num_objects': False,
                                                                    'num_unique_classes': False,
                                                                    'mean_bbox_area': False,
                                                                    'median_bbox_area': False,
                                                                    'max_bbox_area': False,
                                                                    'min_bbox_area': False,
                                                                    'mean_aspect_ratio': False,
                                                                    'mean_conf': False,
                                                                    'median_conf': False,
                                                                    'max_conf': False,
                                                                    'min_conf': False,
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

    stats, _, _ = _bbox_stats(boxes, cls_col=4, nan_default=nan_default)

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


@tensorleap_custom_visualizer("image_with_pred_boxes", LeapDataType.ImageWithBBox)
def image_with_pred_boxes_visualizer(
    image: np.ndarray,
    preds: np.ndarray,
    data: SamplePreprocessResponse,
) -> LeapImageWithBBox:
    """
    Visualize predictions in (xyxy + obj + class scores) format from pre-NMS output.
    """
    if preds.shape[1]==8400:
        boxes = postprocess(torch.tensor(preds), conf_thre=CONF_THRESH, nms_thre=NMS_THRESH,
                        num_classes=NUM_CLASSES, class_agnostic=True)[0]
    else:
        boxes = preds.copy()[0,::]
    meta_data = metadata_image_info_a(int(data.sample_ids), data.preprocess_response)
    img_viz, r = post_process_image(image, meta_data)
    if boxes is None or boxes.size == 0:
        return LeapImageWithBBox(img_viz, [])

    boxes = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    boxes_arr = boxes[:, :4].astype(np.float32)
    cls_ids = boxes[:, -1].astype(int)
    scores_obj = boxes[:, 4] if boxes.shape[1] > 4 else np.ones(len(boxes_arr), dtype=np.float32)

    # Denormalize from resized pixels back to original pixels.
    boxes_arr[:, [0, 2]] /= r
    boxes_arr[:, [1, 3]] /= r

    H, W = img_viz.shape[:2]
    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    leap_boxes = []
    for cls, cxi, cyi, wi, hi, conf in zip(cls_ids, cx, cy, w, h, scores_obj):
        leap_boxes.append(
            BoundingBox(
                x=float(cxi / W),
                y=float(cyi / H),
                width=float(wi / W),
                height=float(hi / H),
                confidence=float(conf),
                label=CLASSES[int(cls)],
            )
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

            # denormalize to original pixels
            gt_xyxy[:, [0, 2]] /= r
            gt_xyxy[:, [1, 3]] /= r

            H, W = img_viz.shape[:2]
            x1, y1, x2, y2 = gt_xyxy[:, 0], gt_xyxy[:, 1], gt_xyxy[:, 2], gt_xyxy[:, 3]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            for cls, cxi, cyi, wi, hi in zip(gt_cls_ids, cx, cy, w, h):
                leap_boxes.append(
                    BoundingBox(
                        x=float(cxi / W),
                        y=float(cyi / H),
                        width=float(wi / W),
                        height=float(hi / H),
                        confidence=1.0,
                        label=CLASSES[int(cls)] +"_gt" ,
                        metadata={"source": "gt"},
                    )
                )

    # ---- Pred boxes ----
    if preds is not None:
        if preds.shape[1] == 8400:
            pred_boxes = postprocess(
                torch.tensor(preds),
                conf_thre=0.3,
                nms_thre=0.45,
                num_classes=NUM_CLASSES,
                class_agnostic=True,
            )[0]
        else:
            pred_boxes = preds.copy()[0, ::]

        if pred_boxes is not None and pred_boxes.size != 0:
            pred_boxes = (
                pred_boxes.numpy()
                if isinstance(pred_boxes, torch.Tensor)
                else pred_boxes
            )
            pred_xyxy = pred_boxes[:, :4].astype(np.float32)
            pred_cls_ids = pred_boxes[:, -1].astype(int)
            pred_scores = pred_boxes[:, 4] if pred_boxes.shape[1] > 4 else np.ones(
                len(pred_xyxy), dtype=np.float32
            )

            # denormalize to original pixels
            pred_xyxy[:, [0, 2]] /= r
            pred_xyxy[:, [1, 3]] /= r

            H, W = img_viz.shape[:2]
            x1, y1, x2, y2 = pred_xyxy[:, 0], pred_xyxy[:, 1], pred_xyxy[:, 2], pred_xyxy[:, 3]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            for cls, cxi, cyi, wi, hi, conf in zip(pred_cls_ids, cx, cy, w, h, pred_scores):
                leap_boxes.append(
                    BoundingBox(
                        x=float(cxi / W),
                        y=float(cyi / H),
                        width=float(wi / W),
                        height=float(hi / H),
                        confidence=float(conf),
                        label=CLASSES[int(cls)] + "_pred",
                        metadata={"source": "pred"},
                    )
                )

    return LeapImageWithBBox(img_viz, leap_boxes)



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


def _match_detections_iou_only(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float,
):
    """Greedy one-to-one matching by IoU only (class-agnostic)."""
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return [], set(), set(range(len(gt_boxes)))

    ious = bboxes_iou(
        torch.from_numpy(pred_boxes[:, :4]),
        torch.from_numpy(gt_boxes[:, :4]),
    ).numpy()
    matches = []
    used_gt = set()
    used_pred = set()
    for p_idx in np.argsort(-ious.max(axis=1)):
        if p_idx in used_pred:
            continue
        gt_idx = int(np.argmax(ious[p_idx]))
        if gt_idx in used_gt:
            continue
        if ious[p_idx, gt_idx] >= iou_thresh:
            matches.append((p_idx, gt_idx))
            used_gt.add(gt_idx)
            used_pred.add(p_idx)
    unused_pred = set(range(len(pred_boxes))) - used_pred
    unused_gt = set(range(len(gt_boxes))) - used_gt
    return matches, unused_pred, unused_gt


@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def ious(preds: np.ndarray, gt_bboxes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Greedy one-to-one IoU matching.
    Returns per-class mean IoU (over GT instances) and mean sample IoU.
    """
    default_value = np.ones(1, dtype=np.float32) * float("nan")
    iou_dic = {cls_name: default_value for cls_name in CLASSES}

    # Decode predictions to xyxy (+ obj + cls) format.
    if preds.shape[1] == 8400:
        decoded = postprocess(
            torch.tensor(preds),
            conf_thre=CONF_THRESH,
            nms_thre=NMS_THRESH,
            num_classes=NUM_CLASSES,
            class_agnostic=True,
        )[0]
        decoded = np.zeros((0, 7), dtype=np.float32) if decoded is None else decoded.cpu().numpy()
    else:
        decoded = preds.copy()[0, ::]

    # Flatten GT to [N, 5] and drop invalid rows.
    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    if gt.size:
        valid_gt = (gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1]) & ~np.isnan(gt[:, 4])
        gt = gt[valid_gt]

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
    if preds.shape[1] == 8400:
        decoded = postprocess(
            torch.tensor(preds),
            conf_thre=0.0,
            nms_thre=NMS_THRESH,
            num_classes=NUM_CLASSES,
            class_agnostic=True,
        )[0]
        decoded = np.zeros((0, 7), dtype=np.float32) if decoded is None else decoded.cpu().numpy()
    else:
        decoded = preds.copy()[0, ::]

    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    if gt.size:
        valid_gt = (gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1]) & ~np.isnan(gt[:, 4])
        gt = gt[valid_gt]

    pred_xyxy = decoded[:, :4] if decoded.size else np.zeros((0, 4), dtype=np.float32)
    pred_obj = decoded[:, 4] if decoded.size else np.zeros((0,), dtype=np.float32)
    keep = pred_obj >= OBJ_THRESH
    pred_xyxy = pred_xyxy[keep]

    gt_xyxy = gt[:, :4] if gt.size else np.zeros((0, 4), dtype=np.float32)

    matches, unused_pred, unused_gt = _match_detections_iou_only(
        pred_xyxy, gt_xyxy, iou_thresh=IOU_THRESH
    )
    tp = float(len(matches))
    fp = float(len(unused_pred))
    fn = float(len(unused_gt))

    precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = tp / (tp + fp + fn + 1e-9) if (tp + fp + fn) > 0 else float("nan")

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
    if preds.shape[1]==8400:
        decoded = postprocess(torch.tensor(preds), conf_thre=CONF_THRESH, nms_thre=NMS_THRESH,
                              num_classes=NUM_CLASSES, class_agnostic=True)[0]
        if decoded is None:
            decoded = np.zeros((0, 7), dtype=np.float32)
        else:
            decoded = decoded.cpu().numpy()
    else:
        decoded = preds.copy()[0,::]

    gt = gt_bboxes if gt_bboxes.ndim == 2 else gt_bboxes.reshape(-1, gt_bboxes.shape[-1])
    pred_boxes = decoded[:, :5] if decoded.size else np.zeros((0, 5), dtype=np.float32)
    pred_cls = decoded[:, -1:] if decoded.size else np.zeros((0, 1), dtype=np.float32)
    pred_boxes = np.concatenate([pred_boxes[:, :4], pred_cls], axis=1) if pred_boxes.size else pred_boxes

    matches, unused_pred, unused_gt = _match_detections(pred_boxes, gt, iou_thresh=DET_IOU_THRESH)
    tp = float(len(matches))
    fp = float(len(unused_pred))
    fn = float(len(unused_gt))

    precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = tp / (tp + fp + fn + 1e-9) if (tp + fp + fn) > 0 else float("nan")

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

        metrics[f"precision_{cls_name}"] = np.array([precision_c], dtype=np.float32)
        metrics[f"recall_{cls_name}"] = np.array([recall_c], dtype=np.float32)
        metrics[f"F1_{cls_name}"] = np.array([f1_c], dtype=np.float32)
        metrics[f"accuracy_{cls_name}"] = np.array([accuracy_c], dtype=np.float32)

    return metrics
