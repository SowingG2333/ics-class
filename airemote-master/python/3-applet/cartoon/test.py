"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""
import cv2 as cv
from inferemote.testing import AiremoteTest
from air import Cartoon
 
class MyTest(AiremoteTest):
    ''' Define a callback function for inferencing, which will be called 
        for every single image '''
    def run(self, image):
        orig_shape = image.shape[:2]
        result = self.air.inference(image) 
        new_image = cv.resize(result, orig_shape[::-1])
        return new_image

if __name__ == '__main__':
    url_image = "https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/data/test.jpg"

    air = Cartoon()
    t = MyTest(air=air, input=url_image, mode='liveweb')

    t.start(remote='tcp://adk:5572', mode='show')

#Ends
