import numpy as np
import torch

from yolox.utils import postprocess


def decode_preds(
    preds: np.ndarray,
    *,
    conf_thre: float,
    nms_thre: float,
    num_classes: int,
    class_agnostic: bool,
) -> np.ndarray:
    """Decode predictions to xyxy (+obj +cls_conf +cls) or return empty array."""
    if preds.shape[1] == 8400:
        decoded = postprocess(
            torch.tensor(preds),
            conf_thre=conf_thre,
            nms_thre=nms_thre,
            num_classes=num_classes,
            class_agnostic=class_agnostic,
        )[0]
        return np.zeros((0, 7), dtype=np.float32) if decoded is None else decoded.cpu().numpy()
    if preds.ndim == 2:
        return preds.copy()
    return preds.copy()[0, ::]
