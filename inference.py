from ultralytics import YOLO

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import tqdm
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import time


def do_the_thing(model_path):
    video = cv2.VideoCapture("../test_vid.mp4")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter(
        "out.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )

    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = model_path
    cfg.DATASETS.TRAIN = ("fishies_train",)
    predictor = DefaultPredictor(cfg)

    v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

    def runOnVideo(video, maxFrames):
        """Runs the predictor on every frame in the video (unless maxFrames is given),
        and returns the frame with the predictions drawn.
        """
        readFrames = 0
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break

            # Get prediction results for this frame
            outputs = predictor(frame)

            # Make sure the frame is colored
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(
                frame, outputs["instances"].to("cpu")
            )

            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            yield visualization

            readFrames += 1
            if readFrames > maxFrames:
                break

    start = time.time()
    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
        cv2.imwrite("POSE detectron2.png", visualization)
        video_writer.write(visualization)
    end = time.time()
    time_taken = end - start

    print(
        f"""
    ***STATISTICS FOR {model_path}***
    
    Time taken: {time_taken}
    Average FPS: {num_frames / time_taken}
    """
    )

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()


def look_once(model_path):
    video = cv2.VideoCapture("../test_vid.mp4")
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    start = time.time()
    model = YOLO(model_path)
    model("../test_vid.mp4")
    end = time.time()
    time_taken = end - start

    print(
        f"""
        ***STATISTICS FOR {model_path}***
    
        Time taken: {time_taken}
        Average FPS: {num_frames / time_taken}
        """
    )
