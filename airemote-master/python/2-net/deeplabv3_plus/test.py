#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
''' Prepare a test '''
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from deeplabv3 import Deeplabv3
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        mask = self.air.inference_remote(image)
        new_img = self.air.make_mask(image, mask)
        return new_img

if __name__ == '__main__':

    air = Deeplabv3()
    t = MyTest(air=air, mode='liveweb', threads=1)

    t.start(remote='adk', input='/Users/haojiash/test/pictures', mode='show')

#Ends
