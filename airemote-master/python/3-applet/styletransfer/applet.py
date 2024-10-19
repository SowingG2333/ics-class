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

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder

class Applet(AtlasApplet):

    def __init__(self):
        super().__init__(port=5550)

    def pre_process(self, blob):
        image = ImageEncoder.bytes_to_image(blob)
        '''Transposing must go here, after jpeg decoding by cv2'''
        image = image.transpose(2, 0, 1).copy()
        image = image.tobytes()

        return image

    def post_process(self, result):
        '''Transposing must go here, before jpeg encoding by cv2'''
        result = result[0]

        ''' This is ** THE BIG DIFFERENCE ** '''
        result = np.frombuffer(result, np.float32)
        ''' Difference Ends '''

        image = result.reshape(3, 360, 540).astype(np.float32)
        image = image.transpose(1, 2, 0).copy()

        blob = ImageEncoder.image_to_bytes(image)
        return [blob]

    def helpinfo(self):
        return "Picasso applet for compressing image data to and from the remote."

if __name__ == '__main__':
    Applet().run()

# Ends.
