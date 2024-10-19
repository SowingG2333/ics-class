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

class Portrait(AtlasRemote):
    '''
    Model : /home/HwHiAiUser/models/portrait/portrait.om
    SOURCE: https://gitee.com/ascend/samples/tree/master/python/contrib/portrait_picture
    '''
    MODEL_WIDTH  = 224
    MODEL_HEIGHT = 224

    def __init__(self, **kwargs):
        super().__init__(port=5567, **kwargs)

        BG_IMG = os.path.dirname(os.path.abspath(__file__))+'/background.jpg'
        if os.path.isfile(BG_IMG):
            self.bg_image = cv.imread(BG_IMG)
        else:
            print('NO background image: {}'.format(BG_IMG))
            self.bg_image = None

    def pre_process(self, bgr_img):
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        image = cv.resize(rgb_img, (224, 224)).astype(np.float32)

        ''' Encode to bytes before push to remote models'''
        blob = image.tobytes()
        return blob

    def post_process(self, result):
        blob = np.frombuffer(result[0], np.float32)
        vals = blob.flatten()
        mask = np.clip((vals * 255), 0, 255)
        mask = mask.reshape(224, 224, 2)
        mask = mask[:, :, 0]

        return mask 

    def make_result(self, orig_img, mask):
        """
        Combine the segmented portrait with the background image
        """
        bg_img = self.bg_image
        height, width = orig_img.shape[:2]
        mask = mask / 255
        mask_resize = cv.resize(mask, (width, height))
        background  = cv.resize(bg_img, (width, height))

        background  = cv.GaussianBlur(background, (15,15), 0)

        mask_bg = np.repeat(mask_resize[..., np.newaxis], 3, 2)
        result  = np.uint8(background * mask_bg + orig_img * (1 - mask_bg))
        return result

# Ends.
