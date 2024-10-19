#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from gesture import Gesture
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        shape = image.shape[:2]
        text = self.air.inference_remote(image)
        new_img = self.air.make_image(text, shape)
        return new_img

if __name__ == '__main__':
    
    air = Gesture()
    url_images = 'https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/gesture_recognition/test_image/test1.jpg;https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/gesture_recognition/test_image/test2.jpg;https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/gesture_recognition/test_image/test3.jpg'

    t = MyTest(air=air, input=url_images, mode='mshow')

    t.start(remote='adk')

#Ends

