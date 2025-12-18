from typing import Tuple
import numpy as np
import torch
from yolox.utils import bboxes_iou


def prf1_from_counts(tp: float, fp: float, fn: float) -> Tuple[float, float, float, float]:
    """Return precision, recall, F1, accuracy with NaN for undefined cases."""
    precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = tp / (tp + fp + fn + 1e-9) if (tp + fp + fn) > 0 else float("nan")
    return precision, recall, f1, accuracy


def match_detections(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    iou_thresh: float,
) -> Tuple[list, set, set]:
    """Greedy one-to-one matching of predicted boxes to GT by IoU and class id."""
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return [], set(), set(range(len(gt_boxes)))

    # IoU matrix between predictions and GT (xyxy).
    ious = bboxes_iou(
        torch.from_numpy(pred_boxes[:, :4]),
        torch.from_numpy(gt_boxes[:, :4]),
    ).numpy()
    matches = []
    used_gt = set()
    used_pred = set()
    # Greedy assignment in descending order of each prediction's best IoU.
    for p_idx in np.argsort(-ious.max(axis=1)):
        if p_idx in used_pred:
            continue
        gt_idx = int(np.argmax(ious[p_idx]))
        if gt_idx in used_gt:
            continue
        # Match only if IoU threshold met and class id matches.
        if ious[p_idx, gt_idx] >= iou_thresh and pred_boxes[p_idx, 4] == gt_boxes[gt_idx, 4]:
            matches.append((p_idx, gt_idx))
            used_gt.add(gt_idx)
            used_pred.add(p_idx)
    unused_pred = set(range(len(pred_boxes))) - used_pred
    unused_gt = set(range(len(gt_boxes))) - used_gt
    return matches, unused_pred, unused_gt


def match_detections_iou_only(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    *,
    iou_thresh: float,
) -> Tuple[list, set, set]:
    """Greedy one-to-one matching by IoU only (class-agnostic)."""
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return [], set(), set(range(len(gt_boxes)))

    # IoU matrix between predictions and GT (xyxy).
    ious = bboxes_iou(
        torch.from_numpy(pred_boxes[:, :4]),
        torch.from_numpy(gt_boxes[:, :4]),
    ).numpy()
    matches = []
    used_gt = set()
    used_pred = set()
    # Greedy assignment in descending order of each prediction's best IoU.
    for p_idx in np.argsort(-ious.max(axis=1)):
        if p_idx in used_pred:
            continue
        gt_idx = int(np.argmax(ious[p_idx]))
        if gt_idx in used_gt:
            continue
        # Match only on IoU threshold (no class check).
        if ious[p_idx, gt_idx] >= iou_thresh:
            matches.append((p_idx, gt_idx))
            used_gt.add(gt_idx)
            used_pred.add(p_idx)
    unused_pred = set(range(len(pred_boxes))) - used_pred
    unused_gt = set(range(len(gt_boxes))) - used_gt
    return matches, unused_pred, unused_gt
