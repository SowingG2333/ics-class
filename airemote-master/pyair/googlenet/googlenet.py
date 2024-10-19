import numpy as np
import cv2 as cv

from .image_net_classes import get_image_net_class
from airlib import AirObject

class GoogleNet(AirObject):
    MODEL_WIDTH  = 224
    MODEL_HEIGHT = 224

    def pre_process(self, bgr_img):
        img = bgr_img
        img = cv.resize(img, (self.MODEL_WIDTH, self.MODEL_HEIGHT))
        blob = img.tobytes()
        return blob

    def post_process(self, result):
        blob = result[0]
        blob = np.frombuffer(blob, np.float32)
        blob = blob.reshape((1, 1000, 1, 1, ))

        vals = blob[0].flatten()
        top_k = vals.argsort()[-1:-6:-1]
        print("======== top5 inference results: =============")
        for n in top_k:
            object_class = get_image_net_class(n)
            print("label:%d  confidence: %f, class: %s" % (n, vals[n], object_class))

        text = get_image_net_class(top_k[0])
        print(f"\nResult: ``{text}''\n")
        print("post_process DONE.")
        return text

''' Ends. '''
