#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from portrait import Portrait
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        mask = self.air.inference_remote(image)
        new_img = self.air.make_result(image, mask)
        return new_img

if __name__ == '__main__':
    
    air = Portrait()
    url_image = 'https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/Portrait/ori.jpg'

    t = MyTest(air=air, input=url_image, mode='show')
    t.start(remote='adk')

#Ends

