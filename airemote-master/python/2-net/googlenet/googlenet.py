#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310
                <https://gitee.com/haojiash/airemote>
"""
import sys, os
import numpy as np
import cv2 as cv

from inferemote.atlas_remote import AtlasRemote
from image_net_classes import get_image_net_class

class GoogleNet(AtlasRemote):
    MODEL_WIDTH  = 224
    MODEL_HEIGHT = 224

    def __init__(self, **kwargs):
        super().__init__(port=5530, **kwargs)

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
        return text

if __name__ == '__main__':
    ''' Running tips  '''
    if (len(sys.argv) != 3):
        print("\n Usage: python {} <remote_string> <image_path> \n\n \
	The remote_string goes like ``192.168.1.123'' or \n\
	``file:///home/HwHiAiUser/models/googlenet/googlenet.om''\n".format(sys.argv[0]))
        sys.exit()
    ''' Only one picture is used '''
    image = cv.imread(sys.argv[2])
    if image is None:
        sys.exit()
    ''' Create an AiRemote object and running throug a REMOTE! '''
    air = GoogleNet()
    air.use_remote(sys.argv[1])
    ''' Run and print the result '''
    text = air.inference_remote(image)
    print(f"\nClass predicted: ``{text}''\n")

''' Ends. '''
