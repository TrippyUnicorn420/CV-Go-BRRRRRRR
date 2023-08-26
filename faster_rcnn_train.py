"""
Make sure you have Detectron2 installed. It is not supported (well) on Windows
but it works well enough.
"""

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import cv2
import random
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

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

my_dataset_train_metadata = MetadataCatalog.get("fishies_train")
dataset_dicts = DatasetCatalog.get("fishies_train")

import random
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 6):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.9
    )
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("image", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("fishies_train",)
cfg.DATASETS.TEST = ("fishies_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
