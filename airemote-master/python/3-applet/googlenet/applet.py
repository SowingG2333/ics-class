#!/usr/bin/env python3
"""
Inferemote: a Remote Inference Toolkit for Ascend 310

Copyright (c) 2021 Jiasheng Hao <haojiash@qq.com>
(University of Electronic Science and Technology of China, UESTC)

Permission  is  hereby  granted,  free  of  charge,  to  any  person
obtaining a copy of this software and associated documentation files
(the  "Software"),  to deal  in  the  Software without  restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to  permit persons to whom  the Software is furnished  to do so,
subject to the following conditions:

The  above copyright  notice  and this  permission  notice shall  be
included in all copies or substantial portions of the Software.

THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,
EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO  THE WARRANTIES OF
MERCHANTABILITY,    FITNESS   FOR    A   PARTICULAR    PURPOSE   AND
NONINFRINGEMENT. IN NO EVENT SHALL  THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER LIABILITY,  WHETHER IN AN
ACTION OF  CONTRACT, TORT OR OTHERWISE,  ARISING FROM, OUT OF  OR IN
CONNECTION WITH  THE SOFTWARE OR  THE USE  OR OTHER DEALINGS  IN THE
SOFTWARE.

"""
import os, sys
import numpy as np
import cv2 as cv
import pickle

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder
from image_net_classes import get_image_net_class

class Applet(AtlasApplet):
    MODEL_WIDTH  = 224
    MODEL_HEIGHT = 224

    def __init__(self):
        super().__init__(port=5530)

    def pre_process(self, blob):

        image = ImageEncoder.bytes_to_image(blob).astype(np.uint8)
        image = cv.resize(image, (self.MODEL_WIDTH, self.MODEL_HEIGHT))
        '''Fixed: bytes should be warranted here '''
        data = image.tobytes()

        return data

    def post_process(self, result):
        blob = result[0]
        ''' *** DIFERENCE HERE ***'''
        blob = np.frombuffer(blob, np.float32)
        blob = blob.reshape((1, 1000, 1, 1, ))
        ''' *** DIFERENCE END ***'''
        vals = blob[0].flatten()
        top_k = vals.argsort()[-1:-6:-1]
        print("======== top5 inference results: =============")
        for n in top_k:
            object_class = get_image_net_class(n)
            print("label:%d  confidence: %f, class: %s" % (n, vals[n], object_class))

        text = get_image_net_class(top_k[0]).encode()

        '''return in a list''' 
        return [text]

    def helpinfo(self):
        return "SAMPLE Applet: GoogleNet Pre && Post processing."

if __name__ == '__main__':
    Applet().run()

# Ends.
