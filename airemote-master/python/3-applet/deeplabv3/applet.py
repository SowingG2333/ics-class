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
import numpy as np
import cv2 as cv

from inferemote.atlas_applet import AtlasApplet
from inferemote.image_encoder import ImageEncoder

class Applet(AtlasApplet):
    MODEL_WIDTH  = 513
    MODEL_HEIGHT = 513
    '''Extra processing may be need before execution by real model '''

    def __init__(self):
        super().__init__(port=5580)

    def pre_process(self, blob):
        img = ImageEncoder.bytes_to_image(blob)
        data = img.astype(np.int8).tobytes()

        return data

    def post_process(self, result):
        result = np.frombuffer(result[0], np.float32)
        image = result.reshape(513, 513, 1)
        blob = ImageEncoder.image_to_bytes(image)

        return [blob]

    def helpinfo(self):
        return "SAMPLE Applet: DeepLabv+ Pre && Post processing."

if __name__ == '__main__':
    Applet().run()


# Ends.
