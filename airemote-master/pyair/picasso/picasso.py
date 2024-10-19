import sys, os
import numpy as np
import cv2 as cv

class Picasso():
    #def __init__(self, **kwargs):
    #    super().__init__(port=5550, **kwargs)

    def pre_process(self, bgr_img):
        print("pre_process START...")
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = cv.resize(rgb_img, (1080, 720))
        rgb_img = rgb_img.transpose(2, 0, 1).copy().astype(np.float32)
        ''' Encode to bytes before push to remote models'''
        print("pre_process DONE.")
        return rgb_img.tobytes()

    def post_process(self, result):
        print("post_process START...")
        blob = np.frombuffer(result[0], np.float32)

        image = blob.reshape(3, 360, 540).astype(np.float32)
        image = image.transpose(1, 2, 0).copy()

        ''' set type to uint8 for VideoWriter '''
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR).astype(np.uint8)

        cv.imwrite('result.jpg', image)
        print("post_process DONE. (result.jpg)")
        return image

''' Ends. '''
