# import the COCO Evaluator to use the COCO Metrics
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def main():
    # register your data
    register_coco_instances(
        "fishies_train",
        {},
        "./d2_fishies/train/_annotations.coco.json",
        "./d2_fishies/train/",
    )
    register_coco_instances(
        "fishies_test",
        {},
        "./d2_fishies/test/_annotations.coco.json",
        "./d2_fishies/test/",
    )
    # load the config file, configure the threshold value, load weights
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "./fully_trained_models/faster_rcnn_deep_360px.pth"

    # Create predictor
    predictor = DefaultPredictor(cfg)

    # Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("fishies_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "fishies_test")

    # Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == "__main__":
    main()
