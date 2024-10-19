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
import sys, os, time
import numpy as np
import cv2 as cv
import pickle, json, zstd

from inferemote.atlas_remote import AtlasRemote
from inferemote.image_encoder import ImageEncoder

class Yolov4(AtlasRemote):
    MODEL_WIDTH  = 608
    MODEL_HEIGHT = 608

    def __init__(self, **kwargs):
        super().__init__(port=5562, **kwargs)

    def pre_process(self, frame):
        ''' A lots of hours spent here for Jpeg-encoding of image before sending to the remote.
            Main changes here include:
            1) simplifying image process just with Opencv, with PIL image operations moving out
            2) reducing dramatically the bytes of image before transportation, which is especially
               important for HTTP remotes in a wider range of network accessing
            3) floating operations of the image should run beside the Jpeg-encoding/decodings, which
               cost a lots of hours for me
	'''
        orig_shape = frame.shape[:2] # ORDER: width, height!
        #print("\norig_shape: {}\n".format(orig_shape))
        ' PIL image was removed'
        #image = Image.fromarray(cv.cvtColor(frame.astype(np.uint8), cv.COLOR_BGR2RGB))

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h = image.shape[0]
        img_w = image.shape[1]
        orig_shape = (img_w, img_h)

        net_h = self.MODEL_HEIGHT
        net_w = self.MODEL_WIDTH

        scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        shift_x = (net_w - new_w) // 2
        shift_y = (net_h - new_h) // 2
        shift_x_ratio = (net_w - new_w) / 2.0 / net_w
        shift_y_ratio = (net_h - new_h) / 2.0 / net_h

        tmp_image = cv.resize(image, (new_w, new_h))

        new_image = np.zeros((net_h, net_w, 3), np.uint8)
        new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(tmp_image)
        
        ''' Compress the image for network transportation '''
        blob = ImageEncoder.image_to_bytes(new_image)

        #####
        # This should be done AFTER the Jpeg Encoding, so move them to Model Side
        #new_image = new_image.astype(np.float32)
        #new_image = new_image / 255

        return blob

    def post_process(self, result):
        result = result[0]
        #result = zstd.decompress(result)
        ''' load to list from bytes'''
        result_list = pickle.loads(result)

        return result_list

    def _post_process(self, result):
        result = result[0]
        result = zstd.decompress(result)
        ''' load to list from bytes'''
        result = pickle.loads(result)

        output0 = np.frombuffer(result[0], np.float32)
        output0 = output0.reshape((1,255,19,19,))
        output1 = np.frombuffer(result[1], np.float32)
        output1 = output1.reshape((1,255,38,38,))
        output2 = np.frombuffer(result[2], np.float32)
        output2 = output2.reshape((1,255,76,76,))
       
        infer_output = [output0, output1, output2]
        #print(infer_output[0].shape)
        #print(infer_output[1].shape)
        #print(infer_output[2].shape)
        
        img_w = self.orig_shape[0]
        img_h = self.orig_shape[1]
        scale = min(float(self.MODEL_WIDTH) / float(img_w), float(self.MODEL_HEIGHT) / float(img_h))
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        shift_x_ratio = (self.MODEL_WIDTH - new_w) / 2.0 / self.MODEL_WIDTH
        shift_y_ratio = (self.MODEL_HEIGHT - new_h) / 2.0 / self.MODEL_HEIGHT
        class_number = len(labels)
        num_channel = 3 * (class_number + 5)
        x_scale = self.MODEL_WIDTH / float(new_w)
        y_scale = self.MODEL_HEIGHT / float(new_h)
        all_boxes = [[] for ix in range(class_number)]
        for ix in range(3):    
            pred = infer_output[ix]
            #print('pred.shape', pred.shape)
            anchors = anchor_list[ix]
            boxes = decode_bbox(pred, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio)
            all_boxes = [all_boxes[iy] + boxes[iy] for iy in range(class_number)]

        res = apply_nms(all_boxes, iou_threshold)
        result_list = {}
        if not res:
            result_list['detection_classes'] = []
            result_list['detection_boxes'] = []
            result_list['detection_scores'] = []
        else:
            new_res = np.array(res)
            picked_boxes = new_res[:, 0:4]
            picked_boxes = picked_boxes[:, [1, 0, 3, 2]]
            picked_classes = convert_labels(new_res[:, 4])
            picked_score = new_res[:, 5]
            result_list['detection_classes'] = picked_classes
            result_list['detection_boxes'] = picked_boxes.tolist()
            result_list['detection_scores'] = picked_score.tolist()
        
        return result_list

''' Ends. '''
