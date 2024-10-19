#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from style_transfer import Picasso

class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        orig_shape = image.shape[:2]
        result = self.air.inference_remote(image)
        new_img = cv.resize(result, orig_shape[::-1])
        return new_img

if __name__ == '__main__':
    url_image = 'https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/data/test.jpg'

    air = Picasso()
    t = MyTest(air=air, input=url_image, mode='liveweb')

    t.start(remote='rpi', input='/Users/haojiash/skiing.mp4', mode='show')

#Ends

