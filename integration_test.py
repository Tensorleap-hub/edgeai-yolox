
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)
from code_loader.plot_functions.visualize import visualize

from leap_binder import *
from yolox.exp import get_exp
import onnxruntime
import os


CKPT_PATH = Path("pretrained_models") / "yolox-s-ti-lite_39p1_57p9_checkpoint.pth"
EXP_FILE = Path("exps") / "default" / "yolox_s_ti_lite.py"
ONNX_PATH = Path("yolox_s_with_pre_nms.onnx")


prediction_type1 = PredictionTypeHandler('output', labels = ["x", "y", "w", "h", "0"], channel_dim=1)
prediction_type2 = PredictionTypeHandler('feat_a', labels=[str(i) for i in range(65)], channel_dim=1)
prediction_type3 = PredictionTypeHandler('feat_b', labels=[str(i) for i in range(65)], channel_dim=1)
prediction_type4 = PredictionTypeHandler('feat_c', labels=[str(i) for i in range(65)], channel_dim=1)

@tensorleap_load_model([prediction_type1, prediction_type2,prediction_type3,prediction_type4])
def load_model():
    exp = get_exp(str(EXP_FILE), None)
    m_path = model_path if model_path != None else 'None_path'
    # validate_supported_models(os.path.basename(cfg.model),m_path)
    if os.path.exists(m_path):
        if m_path.endswith('.onnx'):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
            return sess
        else:
            raise ValueError('Supporting ONNX files only - got {}'.format(m_path))
    else:
        raise FileNotFoundError("Model {} not found".format(model_path))



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
    _ = total_loss(preds[1], preds[2], preds[3], gts)

    #Visualize
    s_prepro = SamplePreprocessResponse(idx, subset)
    image = image_visualizer(img, s_prepro)
    gt_bboxs = image_with_boxes_visualizer(image=img, bboxes=gts, data=s_prepro)
    pred_bboxs = image_with_pred_boxes_visualizer(image=img, preds=preds[0], data=s_prepro)

    # visualize(image)
    # visualize(gt_bboxs)
    # visualize(pred_bboxs)

    meta_data = metadata_image_info_a(idx, subset)
    meta_data2 = metadata_image_info_a(idx, subset)

    metrices = cost(preds[1], preds[2], preds[3], gts)
    stats = detection_prf1(preds[0], gts)


if __name__ == "__main__":
    model_path = '/Users/orram/Tensorleap/edgeai-yolox/yolox_test.onnx'
    # model_path = '/Users/orram/Tensorleap/edgeai-yolox/yolox_s_raw_head_det.onnx'
    datasets: List = preprocess_func()
    sample_subset = datasets[0]
    if sample_subset.length == 0:
        raise RuntimeError("No samples available for integration test")
    for i in range(5):
        322
        ids = np.random.randint(0, 127)
        print(ids)
        check_custom_integration(ids, sample_subset)
        break

