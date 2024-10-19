"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import numpy as np
import cv2 as cv

from inferemote.testing import AiremoteTest
#from lenet import LeNet
from inferemote.airlab import LeNet
 
class MyTest(AiremoteTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ''' An airemote object '''
        self.air = LeNet()

    ''' Define a callback function for inferencing, which will be called for every single image '''
    def run(self, image):
        n = self.air.inference_remote(image)
        shape = image.shape[:2]
        new_image = self.air.make_image(n, shape)
        return new_image


if __name__ == '__main__':
    ''' default image for testing'''
    url_image = 'https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/lenet_mindspore/test_image/test1.png;https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/lenet_mindspore/test_image/test2.png;https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/lenet_mindspore/test_image/test3.png'
    
    t = MyTest(remote='adk')
    t.start(input=url_image, mode='show')

#Ends

