
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)
from code_loader.plot_functions.visualize import visualize

from leap_binder import *
import onnxruntime
import os
import argparse
VISUALIZE = False

prediction_type1 = PredictionTypeHandler('output', labels = ["x0", "y0", "x1", "y1", "conf", "class"], channel_dim=1)
prediction_type2 = PredictionTypeHandler('feat_a', labels=["x0", "y0", "x1", "y1", "c", *[str(i) for i in range(NUM_CLASSES)]], channel_dim=1)
prediction_type3 = PredictionTypeHandler('feat_b', labels=["x0", "y0", "x1", "y1", "c", *[str(i) for i in range(NUM_CLASSES)]], channel_dim=1)
prediction_type4 = PredictionTypeHandler('feat_c', labels=["x0", "y0", "x1", "y1", "c", *[str(i) for i in range(NUM_CLASSES)]], channel_dim=1)

@tensorleap_load_model([prediction_type1, prediction_type2,prediction_type3,prediction_type4])
def load_model():
    m_path = model_path if model_path != None else 'None_path'
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
    tot_loss = total_loss(preds[1], preds[2], preds[3], gts)

    #Visualize
    s_prepro = SamplePreprocessResponse(idx, subset)
    image = image_visualizer(img, s_prepro)
    pred_bboxs = image_with_pred_boxes_visualizer(image=img, preds=preds[0], data=s_prepro)
    comb_bboxs = image_with_gt_and_pred_boxes_visualizer(image=img, bboxes=gts, preds=preds[0], data=s_prepro)
    # present visualizations for testing
    if VISUALIZE:
        visualize(image)
        visualize(comb_bboxs)
        visualize(pred_bboxs)

    meta_data = metadata_image_info_a(idx, subset)

    pred_stats = pred_statistics(preds[0], img, s_prepro)
    metrices = cost(preds[1], preds[2], preds[3], gts)
    stats = detection_prf1(preds[0], gts)
    ious_metric = ious(preds[0], gts)
    obj_prf1 = objectness_prf1(preds[0], gts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensorleap integration smoke test")
    parser.add_argument(
        "--vis-results",
        action="store_true",
        default=True,
        help="Show sample visualizations during the test",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of samples to run (capped by available dataset size)",
    )
    args = parser.parse_args()
    VISUALIZE = args.vis_results
    num_images = max(1, args.num_images)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolox_m_snippet.onnx')

    datasets = preprocess_func()
    sample_subset = datasets[0]
    if sample_subset.length == 0:
        raise RuntimeError("No samples available for integration test")
    for i in range(min(num_images, sample_subset.length, 10)):
        idx = np.random.randint(0, sample_subset.length)
        print(idx)
        check_custom_integration(idx, sample_subset)
