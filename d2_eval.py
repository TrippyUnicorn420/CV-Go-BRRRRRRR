from detectron2.evaluation import COCOEvaluator

from detectron2.data.datasets import register_coco_instances

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

evaluator = COCOEvaluator("fishies_test", distributed=true, output_dir="./output/")
