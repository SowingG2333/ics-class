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
import cv2 as cv
import numpy as np
import pickle

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder

class Applet(AtlasApplet):

    def __init__(self):
        super().__init__(port=5570)

    def pre_process(self, blob):

        image = ImageEncoder.bytes_to_image(blob)
        ''' The following operation must be done outside of jpeg encoding '''
        image = image / 127.5 - 1
        image = image.tobytes()

        return image

    def post_process(self, result):
        blob = np.frombuffer(result[0], np.float32)
        shape = (1, 256, 256, 3, )
        image = blob.reshape(shape)
        ''' The following operation must be done outside of jpeg encoding '''
        image = (np.squeeze(image) + 1) * 127.5
        blob = ImageEncoder.image_to_bytes(image)

        return [blob]


if __name__ == '__main__':
    Applet().run()

# Ends.
