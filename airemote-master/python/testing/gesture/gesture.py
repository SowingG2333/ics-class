"""
Inferemote: a Remote Inference Toolkit for Ascend 310

Base on code from https://gitee.com/ascend/samples.git
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
import sys, os
import numpy as np
import cv2 as cv

from inferemote.atlas_remote import AtlasRemote
from inferemote.image_encoder import ImageEncoder
from gesture_categories import get_gesture_categories

class Gesture(AtlasRemote):
    ''' 
    MODEL  : /home/HwHiAiUser/models/gesture/gesture.om
    SOURCE : https://gitee.com/ascend/samples/tree/master/python/contrib/gesture_recognition_picture
    '''
    MODEL_WIDTH  = 224
    MODEL_HEIGHT = 224

    def __init__(self, **kwargs):
        super().__init__(port=5566, **kwargs)

    def pre_process(self, bgr_img):
        image = cv.resize(bgr_img, (224, 224)).astype(np.float32)
        image = image.transpose(2, 0, 1).copy()

        ''' Encode to bytes before push to remote models'''
        blob = image.tobytes()
  
        return blob

    def post_process(self, result):

        blob = np.frombuffer(result[0], np.float32)
        shape = (1,21,1,1,)
        result = blob.reshape(shape)

        vals = result[0].flatten()
        top_k = vals.argsort()[-1:-2:-1]
        print("======== top5 inference results: =============")
        for n in top_k:
            object_class = get_gesture_categories(n)
            print("label:%d  confidence: %f, class: %s" % (n, vals[n], object_class))

        return get_gesture_categories(top_k[0])

    def make_image(self, text, orig_shape):
        I = np.zeros(orig_shape, dtype=np.uint8)
        image=cv.cvtColor(I, cv.COLOR_GRAY2BGR)
        image=cv.putText(image, text, (50, 100), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        return image

''' Ends. '''
