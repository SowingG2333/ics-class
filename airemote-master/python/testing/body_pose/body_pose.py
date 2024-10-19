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
from pose_decode import decode_pose

class BodyPose(AtlasRemote):
    '''
    Model : OpenPose_light.om, 
    SOURCE: https://gitee.com/ascend/samples/tree/master/python/contrib/body_pose_picture
    '''
    NET_WIDTH  = 368
    NET_HEIGHT = 368

    def __init__(self, **kwargs):
        super().__init__(port=5568, **kwargs)

    def pre_process(self, image):
        sized = cv.resize(image, (self.NET_WIDTH, self.NET_HEIGHT))
        sized = np.asarray(sized, dtype=np.float32) / 255.

        ''' Encode to bytes before push to remote models'''
        blob = sized.tobytes()
  
        return blob

    def post_process(self, result):
        out_shape = (1,14,)
        blob = np.frombuffer(result[0], np.float32)
        heatmaps = blob.reshape(out_shape)

        return heatmaps[0]

    def mark_result(self, image, heatmaps):

        # postprocessing: use the heatmaps (the output of model) to get the joins and limbs for human body
        # Note: the model has multiple outputs, here we used a simplified method, which only uses heatmap for body joints
        #       and the heatmap has shape of [1,14], each value correspond to the position of one of the 14 joints. 
        #       The value is the index in the 92*92 heatmap (flatten to one dimension)
        # calculate the scale of original image over heatmap, Note: image_original.shape[0] is height
        #scale = np.array([image.shape[1] / heatmap_width, image.shape[0]/ heatmap_height])
        canvas = decode_pose(heatmaps, image)
        return canvas

''' Ends. '''
