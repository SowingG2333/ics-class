#!/usr/bin/env python3.9
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
import inferemote.airlab as lab

class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, 
        which will be called for every single image '''
    def run(self, image):
        '''result should be in the same shape as input'''
        return self.air.inference_remote(image)

if __name__ == '__main__':
    air = lab.Load(name='picasso', remote='opi') 
    t = MyTest(air=air, mode='show')
    t.start(input='/Users/haojiash/skiing.mp4', threads=4)
#Ends

