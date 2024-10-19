#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
from inferemote.testing import AiremoteTest
from mark_objects import mark_objects

#from inferemote.airlab import Yolov4
from air import Yolov4
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        bboxes = self.air.inference(image)
        new_image = mark_objects(image, bboxes)
        return new_image

if __name__ == '__main__':
    test_img = 'https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/YOLOV4_coco_detection_car_video/test_video/test.mp4'

    air = Yolov4()
    t = MyTest(air=air, remote='opi', input=test_img, mode='liveweb', threads=4, verbose=True)
    t.start(input='/Users/haojiash/traffic.mp4', mode='show')

#Ends

