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
import yolo.yolov3_postprocessing as postprocessing

class Yolov3(AtlasRemote):
    '''
    Model : /home/HwHiAiUser/models/yolov3/yolov3.om
    SOURCE: https://gitee.com/ascend/samples/tree/master/python/contrib/head_pose_picture
    '''

    NET_WIDTH  = 416
    NET_HEIGHT = 416

    def __init__(self, **kwargs):
        super().__init__(port=5563, **kwargs)

    def pre_process(self, img):

        ih, iw = img.shape[:2]
        # preprocessing: resize and paste input image to a new image with size 416*416
        img = np.array(img, dtype='float32')
        img_resize = cv.resize(img, (self.NET_WIDTH, self.NET_HEIGHT), interpolation=cv.INTER_CUBIC)

        scale = min(self.NET_WIDTH / iw, self.NET_HEIGHT / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        img_new = np.ones((416, 416, 3), np.float32) * 128
        img_new[(self.NET_HEIGHT - nh) // 2: ((self.NET_HEIGHT - nh) // 2 + nh),
                (self.NET_WIDTH - nw) // 2: (self.NET_WIDTH - nw) // 2 + nw, :] = img_resize[(self.NET_HEIGHT - nh) // 2: ((self.NET_HEIGHT - nh) // 2 + nh),
                (self.NET_WIDTH - nw) // 2: (self.NET_WIDTH - nw) // 2 + nw, :]
        
        img_new = img_new / 255.
        ''' Encode to bytes before push to remote models'''
        blob = img_new.tobytes()
        return blob

    def post_process(self, result):
        """" yolo_outputs: output (3 feature maps) of YOLO V3 model, sizes are 
             1*13*13*18; 1*26*26*18; 1*52*52*18 seperately """
        output0 = np.frombuffer(result[0], np.float32)
        output0 = output0.reshape((1,13,13,18,))
        output1 = np.frombuffer(result[1], np.float32)
        output1 = output1.reshape((1,26,26,18,))
        output2 = np.frombuffer(result[2], np.float32)
        output2 = output2.reshape((1,52,52,18,))
        infer_output = [output0, output1, output2]
        
        return infer_output

    def get_boxes(self, out_list, origin_img):
        origin_shape = origin_img.shape[:2]
        # convert yolo output to box axis and score
        box_axis, box_score = postprocessing.yolo_eval(
            out_list, self.get_anchors(), 1, origin_shape)
        # get the crop image and corresponding width/heigh info for WHENet
        nparryList, boxList = postprocessing.get_box_img(origin_img, box_axis)

        return nparryList, boxList

    def get_anchors(self):
        """return anchors

        Returns:
            [ndarray]: anchors array
        """
        SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
        anchors_path = os.path.join(SRC_PATH, './yolo/yolo_anchors.txt')
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

''' Ends. '''
