from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2
import onnxruntime as ort
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)
from code_loader.plot_functions.visualize import visualize

from leap_binder import *
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess

import onnxruntime
import os


CKPT_PATH = Path("pretrained_models") / "yolox-s-ti-lite_39p1_57p9_checkpoint.pth"
EXP_FILE = Path("exps") / "default" / "yolox_s_ti_lite.py"
ONNX_PATH = Path("yolox_s_with_pre_nms.onnx")


prediction_type0 = PredictionTypeHandler('output', labels = ["x", "y", "w", "h", "0"] + list(COCO_CLASSES), channel_dim=1)
prediction_type1 = PredictionTypeHandler("bboxes", labels=list(COCO_CLASSES), channel_dim=1)


@tensorleap_load_model([prediction_type0, prediction_type1])
def load_model():
    exp = get_exp(str(EXP_FILE), None)
    m_path = model_path if model_path != None else 'None_path'
    # validate_supported_models(os.path.basename(cfg.model),m_path)
    if os.path.exists(m_path):
        if m_path.endswith('.onnx'):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
            return sess
        elif m_path.endswith('.h5'):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            raise ValueError('Supporting ONNX and H5 files only - got {}'.format(m_path))
    else:
        raise FileNotFoundError("Model {} not found".format(model_path))


def _run_inference(model: torch.nn.Module, img_path: Path, exp) -> np.ndarray:
    raw = cv2.imread(str(img_path))
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    preproc = ValTransform(legacy=False)
    img, _ = preproc(raw, None, exp.test_size)
    tensor = torch.from_numpy(img).unsqueeze(0).float()
    with torch.no_grad():
        outputs = model(tensor)
        outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True)
    return outputs[0].cpu().numpy() if outputs and outputs[0] is not None else np.zeros((0, 7), dtype=np.float32)


@tensorleap_integration_test()
def check_custom_integration(idx: int, subset):
    #load model
    model = load_model()
    #load input and GT
    img = input_encoder(idx, subset)
    inputs = {'images':img}
    gts = gt_encoder(idx, subset)
    #predict
    preds = model.run(None,inputs)
    #get loss
    _ = yolox_total_loss(preds[0], preds[1], gts)
    #Visualize
    gt_bboxs = image_with_boxes_visualizer(img, gts)

    visualize(gt_bboxs)
    # Trigger custom loss validation (Tensorleap expects this to be registered)

    # return predictions aligned with the registered prediction type
    return preds


if __name__ == "__main__":
    model_path = '/Users/orram/Tensorleap/edgeai-yolox/yolox_s_with_pre_nms.onnx'
    datasets: List = preprocess_func()
    sample_subset = datasets[0]
    if sample_subset.length == 0:
        raise RuntimeError("No samples available for integration test")
    check_custom_integration(0, sample_subset)
