#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
''' Prepare a test '''
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from air import Picasso

class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        ''' an image must be returned in the same shape '''
        # new_img = self.air.inference_remote(image)
        orig_shape = image.shape[:2]
        result = self.air.inference_remote(image)
        new_img = cv.resize(result, orig_shape[::-1])
        return new_img

if __name__ == '__main__':

    air = Picasso()
    t = MyTest(air=air, remote='opi', mode='liveweb', threads=4)

    t.start(input='/Users/sowingg/Downloads/airemote-master/airemote.jpg', mode='show')

#Ends

