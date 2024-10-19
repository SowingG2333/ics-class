#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
''' Prepare a test '''
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from air import GoogleNet
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        orig_shape = image.shape[:2]

        text = self.air.inference_remote(image)

        I = np.zeros(orig_shape, dtype=np.uint8)
        new_img = cv.cvtColor(I,cv.COLOR_GRAY2BGR)
        new_img = cv.putText(new_img, text, (50, 100), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        
        return new_img

if __name__ == '__main__':
    air = GoogleNet()
    t = MyTest(air=air, mode='liveweb', threads=1)

    t.start(remote='opi', input='/Users/haojiash/test/pictures', mode='show')

#Ends

