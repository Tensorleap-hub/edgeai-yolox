from typing import Dict, Tuple
from typing import Dict, List, Tuple
import numpy as np
from code_loader.contract.responsedataclasses import BoundingBox

EPS = 1e-9


def filter_valid_gt(gt: np.ndarray) -> np.ndarray:
    """Filter GT to valid xyxy rows with non-NaN class."""
    if gt.size == 0:
        return gt
    valid_gt = (gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1]) & ~np.isnan(gt[:, 4])
    return gt[valid_gt]


def bbox_stats(
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


def xyxy_to_bounding_boxes(
    xyxy: np.ndarray,
    *,
    cls_ids: np.ndarray,
    scores: np.ndarray,
    r: float,
    image_shape: Tuple[int, int],
    labels: List[str],
    label_suffix: str = "",
    metadata: Dict[str, str] = None,
) -> List[BoundingBox]:
    """Convert xyxy boxes (resized pixels) to BoundingBox list in normalized center coords."""
    if xyxy.size == 0:
        return []

    boxes_arr = xyxy.astype(np.float32)
    boxes_arr[:, [0, 2]] /= r
    boxes_arr[:, [1, 3]] /= r

    H, W = image_shape
    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    md = metadata or {}
    out = []
    for cls, cxi, cyi, wi, hi, conf in zip(cls_ids, cx, cy, w, h, scores):
        out.append(
            BoundingBox(
                x=float(cxi / W),
                y=float(cyi / H),
                width=float(wi / W),
                height=float(hi / H),
                confidence=float(conf),
                label=labels[int(cls)] + label_suffix,
                metadata=md,
            )
        )
    return out
