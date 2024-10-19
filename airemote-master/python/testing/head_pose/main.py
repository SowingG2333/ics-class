"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import cv2 as cv
import os
import copy
import argparse
import sys

from yolov3 import  Yolov3
from whenet import  WHENet

#import logging
#from inferemote.logging import logger
yolo = Yolov3(remote='adk')
whenet = WHENet(remote='adk')

def main(): 
    """main"""
    description = 'head pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', type=str, default='./data/test.jpg', help="Directory path for image")
    args = parser.parse_args()
    
    ''' Run and print the result '''
    print("Loading images from: ", args.input)
    ds = ImageLoader.get_stream(args.input)
    job = InferenceJob(test_func, ds, threads=1, wait=0.001)
    job.submit()

    ''' Processing the results '''
    n = 0
    total = job.get_length()
    while True:
        success, result, name = job.get_result()
        print(success, name)
        if success:
            show_result(result)
        else:
            break

        n += 1
        print("Job running: ", n, "/", total)

''' Read inputs from data source '''
from inferemote.testing.image_loader import ImageLoader

''' for an inference task '''
from inferemote.inference_job import InferenceJob

''' Helper functions '''
def show_result(image):
    from  matplotlib import pyplot as plt
    image = image[:,:,::-1]         # transform image to rgb
    plt.imshow(image)
    plt.show(block=False) 
    plt.pause(1.005)

''' Callback function '''
def test_func(image):
    # convert image to RGB 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # yolov3: inferencing
    result = yolo.inference(image)

    nparryList, boxList  = yolo.get_boxes(result, image)
    if not len(nparryList):
        return image

    # plot yolo and whenet detection result on the image

    image_res = copy.deepcopy(image)
    # whenet: preprocessing, transpose, inference, postprocessing
    detection_result_list = []

    for i in range(len(nparryList)):
        box_width, box_height = (boxList[i][0]+boxList[i][1]) / 2, (boxList[i][2] + boxList[i][3]) / 2
        result = whenet.inference(nparryList[i])
        detection_item = whenet.get_draws(result, box_width, box_height)

        if True:
            #plot head detection box from yolo predictions
            cv.rectangle(image_res, (boxList[i][0], boxList[i][2]), 
                (boxList[i][1], boxList[i][3]), (127, 125, 125), 2)
            #plot head pose detection lines from whenet predictions
            cv.line(image_res, (int(box_width), int(box_height)), 
                (int(detection_item["yaw_x"]), int(detection_item["yaw_y"])), (255, 0, 0), 4)
            cv.line(image_res, (int(box_width), int(box_height)), 
                (int(detection_item["pitch_x"]), int(detection_item["pitch_y"])), (0, 255, 0), 4)
            cv.line(image_res, (int(box_width), int(box_height)), 
                (int(detection_item["roll_x"]), int(detection_item["roll_y"])), (0, 0, 255), 4)

        #image_res = cv.cvtColor(image_res, cv.COLOR_RGB2BGR)
        return image_res

if __name__ == "__main__":
    #logger.setLevel(logging.DEBUG)
    main()
