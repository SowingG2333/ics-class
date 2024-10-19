#!/usr/bin/env python3
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
import sys, os, time, base64
import numpy as np
import cv2 as cv
import pickle

from inferemote.atlas_remote import AtlasRemote
from inferemote.image_encoder import ImageEncoder

class GoogleNet(AtlasRemote):

    def __init__(self, **kwargs):
        super().__init__(port=5532, **kwargs)

    def pre_process(self, img):

        blob = ImageEncoder.image_to_bytes(img)

        return blob

    def post_process(self, result):
        blob = result[0]
        text = blob.decode()
        #print("\n Class predicted: {}\n".format(text))
        return text


if __name__ == '__main__':
    ''' Prepare a test '''
    from inferemote.testing import Testing
    Testing.defaults['mode'] = 'show'
    Testing.defaults['input'] = 'pictures'
    test, remote = Testing.create()

    air = GoogleNet()

    remote = 'tcp://192.168.1.123:5559'
    air.use_remote(remote)

    ''' Define a callback function for testing '''
    def test_func(image):
        orig_shape = image.shape[:2]

        text = air.inference_remote(image)

        I = np.zeros(orig_shape, dtype=np.uint8)
        tmp_image=cv.cvtColor(I,cv.COLOR_GRAY2BGR)
        tmp_image=cv.putText(tmp_image, text, (50, 100), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
   
        return tmp_image

    ''' Start the test '''
    test.run(test_func)
 
''' Ends. '''
