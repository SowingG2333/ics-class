#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import numpy as np
import cv2 as cv
from inferemote.testing import AiremoteTest
from body_pose import BodyPose
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        heatmaps = self.air.inference_remote(image)
        new_img = self.air.mark_result(image.copy(), heatmaps)
        return new_img

if __name__ == '__main__':
    
    air = BodyPose()
    url_images = 'https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/body_pose_picture/test.jpg'

    t = MyTest(air=air, input="camera", mode='show')
    t.start(remote='opi')

#Ends

