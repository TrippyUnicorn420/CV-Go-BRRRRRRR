import torch
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval


data_dir = "./d2_fishies/"
annotations = data_dir + "valid/_annotations.coco.json"
image_dir = data_dir + "valid/"
batch_size = 8
transform = transforms.Compose([transforms.ToTensor()])

valid_dataset = CocoDetection(root=image_dir, annFile=annotations, transform=transform)
dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

num_classes = len(valid_dataset.coco.cats)
coco_gt = valid_dataset.coco

coco_eval = COCOeval(coco_gt, iouType="bbox")

preds = []

with torch.no_grad():
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        predictions = model(images)
        preds.extend(predictions)


coco_eval_preds = []
for prediction in preds:
    boxes = prediction["boxes"].cuda().numpy()
    scores = prediction["scores"].cuda().numpy()
    labels = prediction["labels"].cuda().numpy()

    for box, score, label in zip(boxes, scores, labels):
        coco_eval_preds.append(
            {
                "image_id": 0,
                "category_id": label,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "score": score,
            }
        )

coco_eval.cocoDt = coco_gt.loadRes(coco_eval_preds)

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
