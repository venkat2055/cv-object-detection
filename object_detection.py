import time
from typing import Any, Tuple
import cv2 as cv2
import numpy as np

import argparse
import os
import uuid
from datetime import datetime

"""
This script reads the video file, looks for objects and if found, write the image of the frame 
with bounding boxes

python object_detection.py --input-file-path /Users/Periyasamy/Desktop/test_videos/car_highway_high.mp4 \
                     --output-dir /Users/Periyasamy/Desktop/test_videos/new_output \
                     --model-weights-path 'mobilenet_model/frozen_inference_graph.pb' \
                     --model-graph-path 'mobilenet_model/graph.pbtxt' \
                     --sampling-rate 50
"""

MAX_N_OBJECTS = 3

CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MODEL_CONFIDENCE_THRESHOLd = 0.5


def mobilenetv3_inference(input_img: np.ndarray, model: Any) -> Tuple[np.ndarray]:
    """
    A function that takes an image and a mobilenet model and return outputs of the model
    img: str to the path
    model: loaded model
    """
    # Define the input
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    # Forward pass
    classes_id, confidences, boxes = model.detect(input_img, confThreshold=MODEL_CONFIDENCE_THRESHOLd)

    if len(classes_id):
        classes_id = classes_id.flatten()
        confidences = confidences.flatten()
    return classes_id, confidences, boxes


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img: np.ndarray, class_id: np.ndarray, confidence: np.ndarray, box: np.ndarray) -> None:
    # Loop over detected boxes
    counter = 0
    for c_id, cfd, b in zip(class_id, confidence, box):

        if counter > MAX_N_OBJECTS:
            break

        label = str(CLASSES[c_id])

        conf = '%.2f' % cfd
        # Get the label for the class name and its confidence
        text = '%s:%s' % (label, conf)

        color = COLORS[c_id-1]
        cv2.rectangle(img, b, color, 2)

        font_scale = 1
        labelSize, baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
        top = max(b[1], labelSize[1])
        cv2.putText(img, text, (b[0], top), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 3)
        counter += 1


def main(input_file_path: str, output_dir: str, sampling_rate: int, model_weights_path: str,
         model_graph_path: str) -> None:

    # Load pretrained weights
    net = cv2.dnn_DetectionModel(model_weights_path,
                                 model_graph_path)

    # set opencv backend and CPU inference
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print(f'Input file path: {input_file_path}')

    # check input file exists
    isExist = os.path.exists(input_file_path)
    if not isExist:
        raise ValueError('Input file not found')

    print(f'output dir path: {output_dir}')
    # check if output dir exists
    isExist = os.path.exists(output_dir)

    if not isExist:
        # Create output directory as it does not exist
        print("creating output directory")
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_file_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = 0
    while cap.isOpened():
        success, frame = cap.read()

        if frame is None:
            # stop at end of the video
            break

        # Sampling every n frames
        if success and counter % sampling_rate == 0:

            classe_id, confidence, box = mobilenetv3_inference(frame, net)

            if classe_id is not None and len(classe_id) > 0:
                draw_bounding_box(frame, classe_id, confidence, box)
                image_id = str(uuid.uuid1())
                cv2.imwrite(f'{output_dir}/{image_id}.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='plate detection')

    parser.add_argument('-input-file-path', '--input-file-path', dest='input_file_path', type=str, required=True,
                        help='Path for the video')
    parser.add_argument('-output-dir', '--output-dir', dest='output_dir', type=str, required=True,
                        help='dir for the outputs')
    parser.add_argument('-model-weights-path', '--model-weights-path', dest='model_weights_path', type=str,
                        required=True, help='path model weights')
    parser.add_argument('-model-graph-path', '--model-graph-path', dest='model_graph_path', type=str, required=True,
                        help='path for model graph')
    parser.add_argument('-sampling-rate', '--sampling-rate', dest='sampling_rate', type=int, default=25,
                        help='Sampling every n frames')

    args = parser.parse_args()

    job_start = datetime.now()
    main(input_file_path=args.input_file_path,
         output_dir=args.output_dir,
         model_weights_path=args.model_weights_path,
         model_graph_path=args.model_graph_path,
         sampling_rate=args.sampling_rate, )

    print(f'Total time: {round((datetime.now() - job_start).seconds / 60, 2)} minutes')

