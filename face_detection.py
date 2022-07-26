import time
from typing import Any, Tuple
import cv2 as cv2
import numpy as np

import pandas as pd
import argparse
import os
import uuid
from datetime import datetime
import face_recognition

"""
This script reads the video file, looks for license plates and if found, write the image of the frame 
with bounding boxes and also a csv with list of plates detected and their corresponding images

python face_detection.py --input-file-path /Users/Periyasamy/Desktop/test_videos/famous_celebs.mov \
                           --known-persons-dir '/Users/Periyasamy/Desktop/test_videos/known_persons' \
                           --output-dir /Users/Periyasamy/Desktop/test_videos/new_output \
                           --sampling-rate 50
                     
"""


def get_known_face_encodings(known_faces_dir):
    isExist = os.path.exists(known_faces_dir)
    if not isExist:
        raise ('Known persons dir not found')
    print('getting known person encodings')
    know_person_encodings = []
    know_persons = []
    for img_path in os.listdir(known_faces_dir):
        if img_path[0] == '.':
            # skip os files
            continue

        full_path = os.path.join(known_faces_dir, img_path)
        img = face_recognition.load_image_file(full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        know_person_encodings.append(face_recognition.face_encodings(img_rgb)[0])
        know_persons.append(img_path.split('.')[0])

    return know_person_encodings, know_persons


def recognize_faces(input_img, known_person_encodings, known_persons):
    input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the unknown image
    input_face_locations = face_recognition.face_locations(input_rgb)
    input_face_encodings = face_recognition.face_encodings(input_rgb, input_face_locations)

    best_match_index = -1
    res = None

    for (top, right, bottom, left), face_encoding in zip(input_face_locations, input_face_encodings):
        matches = face_recognition.compare_faces(known_person_encodings, face_encoding)

        if not len(matches):
            continue

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_person_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_persons[best_match_index]
            is_match_found = True

            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(input_rgb, text=name, org=(left + 6, bottom - 2 - 5), fontFace=font, fontScale=1,
                        color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(input_rgb, (left, top), (right, bottom), (0, 255, 0), 3)
            return best_match_index, res


def main(input_file_path: str, output_dir: str, sampling_rate: int, known_persons_dir: str) -> None:

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

    known_person_encodings, known_persons = get_known_face_encodings(known_faces_dir=known_persons_dir)

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

            resp = recognize_faces(frame, known_person_encodings, known_persons)
            if resp is not None:
                best_match_index, out_img = resp[0], resp[1]
                cv2.imwrite(f'{output_dir}/{known_persons[best_match_index]}_{counter}.png', out_img)

        counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='plate detection')

    parser.add_argument('-input-file-path', '--input-file-path', dest='input_file_path', type=str, required=True,
                        help='Path for the video')
    parser.add_argument('-known-persons-dir', '--known-persons-dir', dest='known_persons_dir', type=str, required=True,
                        help='dir with images of the known persons with the name of the person as filename')
    parser.add_argument('-output-dir', '--output-dir', dest='output_dir', type=str, required=True,
                        help='dir for the outputs')
    parser.add_argument('-sampling-rate', '--sampling-rate', dest='sampling_rate', type=int, default=25,
                        help='Sampling every n frames')


    args = parser.parse_args()

    job_start = datetime.now()
    main(input_file_path=args.input_file_path,
         output_dir=args.output_dir,
         sampling_rate=args.sampling_rate,
         known_persons_dir=args.known_persons_dir)

    print(f'Total time: {round((datetime.now() - job_start).seconds / 60, 2)} minutes')

