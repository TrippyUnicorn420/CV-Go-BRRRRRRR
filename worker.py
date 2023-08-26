import inference
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
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
    # if it is commented out, we already have those results
    # print("Faster R-CNN: Fullsize")
    # inference.do_the_thing("./fully_trained_models/faster_rcnn_fullsize.pth")
    # print("Faster R-CNN: 640px")
    # inference.do_the_thing("./fully_trained_models/faster_rcnn_640px.pth")
    # print("RetinaNet: Fullsize")
    # inference.do_the_thing("./fully_trained_models/retinanet_fullsize.pth")
    # print("RetinaNet: 640px")
    # inference.do_the_thing("./fully_trained_models/retinanet_640px.pth")
    # print("YOLO: Fullsize")
    # inference.look_once("./fully_trained_models/yolo_fullsize.pt")
    # print("YOLO: 640px")
    # inference.look_once("./fully_trained_models/yolo_640px.pt")
    # print("YOLO: 480px")
    # inference.look_once("./fully_trained_models/yolo_480px.pt")
    # print("YOLO: 360px")
    # inference.look_once("./fully_trained_models/yolo_360px.pt")
