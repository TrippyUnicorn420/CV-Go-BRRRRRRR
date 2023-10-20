# Automated Real-time Detection of Aquatic Wildlife: Contrasting Speed and Accuracy

This repository contains all the code used to obtain the results shown in my honours thesis with the same title.

## How To Use

Make sure you have a CUDA-enabled version of torch, torchvision and torchaudio installed in your venv. Then, install all the packages as directed in requirements.txt.
Build Detectron2 as follows:
	
	pip install 'git+https://github.com/facebookresearch/detectron2.git'
	


## Training Detectron2 and YOLO

Put a COCO-formatted dataset into a directory called "d2_fishies". Then, run either faster_rcnn_train.py or retinanet_train.py.  
For YOLO, put a YOLOv8-formatted dataset into a directory called "yolo_fishies", and give it an absolute path to `data.yaml` from the root of your drive.

## Benchmarking

Run worker.py. do_the_thing() performs inference on either a Faster R-CNN or a RetinaNet model. look_once() performs inference on a YOLOv8 model.


