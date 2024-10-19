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

class LeNet(AtlasRemote):
    MODEL_WIDTH  = 32
    MODEL_HEIGHT = 32

    def pre_process(self, bgr_img):

        img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        #normalization
        gray_img = gray_img / 255.0
        gray_img = gray_img / 0.3081
        gray_img = gray_img - 1 * 0.1307 / 0.3081

        image = cv.resize(gray_img, (self.MODEL_WIDTH, self.MODEL_HEIGHT)).astype(np.float32)
        
        return image.tobytes()

    def post_process(self, result):
        ''' Decode to image after net travelling '''
        blob = np.frombuffer(result[0], np.float32)

        vals = blob.flatten()
        max_val=np.max(vals)
        vals = np.exp(vals - max_val)
        sum_val = np.sum(vals) 
        vals /= sum_val
        top_k = vals.argsort()[-1:-6:-1]

        return top_k[0]

    def make_image(self, n, orig_shape):
        text = str(n)
        print('NUMBER: ', text)

        I = np.zeros(orig_shape, dtype=np.uint8)
        image=cv.cvtColor(I,cv.COLOR_GRAY2BGR)
        image=cv.putText(image, text, (50, 100), cv.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        return image
    
''' Ends. '''
