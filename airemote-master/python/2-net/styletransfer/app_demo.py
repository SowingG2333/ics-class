#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Atlas 200DK

"""
import os, sys
import cv2 as cv
import numpy as np

''' Read inputs from data source '''
from inferemote.testing.image_loader import ImageLoader

''' New an AiRemote object'''
from style_transfer import StyleTransfer

''' for an inference task '''
from inferemote.inference_job import InferenceJob

''' Helper functions '''
def show_result(image):
    from  matplotlib import pyplot as plt
    image = image [:,:,::-1]         # transform image to rgb
    plt.imshow(image)
    plt.show(block=False) 
    plt.pause(0.005)

''' Callback function '''
def test_func(image):
    orig_shape = image.shape[:2]
    result = air.inference_remote(image)
    result = cv.resize(result, orig_shape[::-1])
    image = np.hstack([image, result])
    return result

if __name__ == '__main__':
    ''' Running tips  '''
    if (len(sys.argv) != 3):
        print("\n Usage: python {} <remote_string> <image_path|image_dir> \n\n The remote_string goes like `192.168.1.123''\n".format(sys.argv[0]))
        sys.exit()

    if not os.path.exists(sys.argv[2]):
        sys.exit()
    ''' Create an AiRemote object and running throug a REMOTE! '''
    air = StyleTransfer()
    rc = air.use_remote(sys.argv[1])
    if not rc:
        sys.exit()

    ''' Run and print the result '''
    ds = ImageLoader.get_stream(sys.argv[2])
    job = InferenceJob(test_func, ds, threads=1, wait=0.01)
    job.submit()

    ''' Processing the results '''
    n = 0
    total = job.get_length()
    while True:
        success, result, name = job.get_result()

        if name:
            show_result(result)
        else:
           break
        n += 1
        print("Job running: ", n, "/", total)

# Ends.
