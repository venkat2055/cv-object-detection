import os
import time
import re
import argparse
from typing import Tuple
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
import cv2
import imutils
import easyocr

"""
This script reads the video file, looks for license plates and if found, write the image of the frame 
with bounding boxes and also a csv with list of plates detected and their corresponding images

python license_plate_detection.py --input-file-path test_videos/car_highway_high.mp4 \
                     --output-dir test_videos/output \
                     --sampling-rate 25
"""

MIN_PLATE_LEN = 5  # Minimum str length for detecting a plate
MAX_PLATE_LEN = 8  # Maximum str length for detecting a plate

OUTPUT_CSV_NAME = 'data.csv'  # filename for csv output


def detect_plate(frame: np.ndarray) -> Tuple[str, np.ndarray]:
    cv2_im = frame.copy()

    gray = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2GRAY)

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)

    if location is not None:

        cv2.drawContours(mask, [location], 0, 255, -1)
        cv2.bitwise_and(cv2_im, cv2_im, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if result:
            text = result[int(len(result)/2)][-2]
            pattern = re.compile('\W')
            text = re.sub(pattern, '', text)

            if MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN:

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv2_im, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
                            color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                res = cv2.rectangle(cv2_im, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

                return text, cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


def main(input_file_path: str, output_dir: str, sampling_rate: int) -> None:

    df = None

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

            resp = detect_plate(frame)

            if resp is not None:
                text, out_image = resp[0], resp[1]

                # generate a random uuid for the file name
                image_id = str(uuid.uuid1())
                cv2.imwrite(f'{output_dir}/{image_id}.png', cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))

                # Data is stored in CSV file
                # timestamp is when the plate was detected and not the timestamp of the frame
                raw_data = {'timestamp': [time.asctime(time.localtime(time.time()))],
                            'v_number': [text],
                            'image_id': [image_id]}

                print(raw_data)
                if df is None:
                    df = pd.DataFrame(raw_data, columns=['timestamp', 'v_number', 'image_id'])
                else:
                    df = pd.concat([df, pd.DataFrame(raw_data, columns=['timestamp', 'v_number', 'image_id'])],
                                   ignore_index=True)

        counter += 1

    if df is not None:
        df.to_csv(f'{output_dir}/{OUTPUT_CSV_NAME}', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='plate detection')

    parser.add_argument('-input-file-path', '--input-file-path', dest='input_file_path', type=str, required=True,
                        help='Path for the video')
    parser.add_argument('-output-dir', '--output-dir', dest='output_dir', type=str, required=True,
                        help='dir for the outputs')
    parser.add_argument('-sampling-rate', '--sampling-rate', dest='sampling_rate', type=int, default=25,
                        help='Sampling every n frames')

    args = parser.parse_args()

    job_start = datetime.now()
    main(input_file_path=args.input_file_path,
         output_dir=args.output_dir,
         sampling_rate=args.sampling_rate, )

    print(f'Total time: {round((datetime.now() - job_start).seconds / 60, 2)} minutes')
