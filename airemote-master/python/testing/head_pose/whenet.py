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
import math

from inferemote.atlas_remote import AtlasRemote

class WHENet(AtlasRemote):
    '''
    Model : /home/HwHiAiUser/models/whenet/WHENet_b2_a1_modified.om
    SOURCE: https://gitee.com/ascend/samples/tree/master/python/contrib/head_pose_picture
    '''
    NET_WIDTH  = 224
    NET_HEIGHT = 224

    def __init__(self, **kwargs):
        super().__init__(port=5564, **kwargs)

    def pre_process(self, image):
        """head pose estimation preprocessing"""
        # convert image to float type
        image = image.astype('float32')
        image = image / 255.
        img_new = cv.resize(image, (224, 224))

        ''' Encode to bytes before push to remote models'''
        blob = img_new.tobytes()
        return blob

    def post_process(self, result):
        
        output0 = np.frombuffer(result[0], np.float32)
        output0 = output0.reshape((1,120,))
        output1 = np.frombuffer(result[1], np.float32)
        output1 = output1.reshape((1,66,))
        output2 = np.frombuffer(result[2], np.float32)
        output2 = output2.reshape((1,66,))
        infer_output = [output0, output1, output2]
        
        return infer_output
    
    def get_draws(self, infer_output, box_width, box_height):
    # postprocessing: convert model output to yaw pitch roll value
        yaw, pitch, roll = self.whenet_angle(infer_output)
        print('Yaw, pitch, roll angles: ', yaw, pitch, roll)
        # obtain coordinate points from head pose angles for plotting
        return self.whenet_draw(yaw, pitch, roll, 
                                tdx=box_width, tdy=box_height, size=200)
    
    def softmax(self, x):
        """softmax"""
        x -= np.max(x, axis=1, keepdims=True)
        a = np.exp(x)
        b = np.sum(np.exp(x), axis=1, keepdims=True)
        return a / b

    def whenet_angle(self, infer_output):
        """
        Obtain yaw pitch roll value in degree based on the output of model

        Args:
            resultList_whenet: result of WHENet

        Returns:
            yaw_predicted, pitch_predicted, roll_predicted: yaw pitch roll values
        """
        yaw = infer_output[0]
        yaw = np.reshape(yaw, (1, 120, 1, 1))
        yaw_out = np.transpose(yaw, (0, 2, 3, 1)).copy()
        yaw_out = yaw_out.squeeze()
        yaw_out = np.expand_dims(yaw_out, axis=0)

        pitch = infer_output[1]
        pitch = np.reshape(pitch, (1, 66, 1, 1))
        pitch_out = np.transpose(pitch, (0, 2, 3, 1)).copy()
        pitch_out = pitch_out.squeeze()
        pitch_out = np.expand_dims(pitch_out, axis=0)

        roll = infer_output[2]
        roll = np.reshape(roll, (1, 66, 1, 1))
        roll_out = np.transpose(roll, (0, 2, 3, 1)).copy()
        roll_out = roll_out.squeeze()
        roll_out = np.expand_dims(roll_out, axis=0)

        yaw_predicted = self.softmax(yaw_out)
        pitch_predicted = self.softmax(pitch_out)
        roll_predicted = self.softmax(roll_out)

        idx_tensor_yaw = [idx for idx in range(120)]
        idx_tensor_yaw = np.array(idx_tensor_yaw, dtype=np.float32)

        idx_tensor = [idx for idx in range(66)]
        idx_tensor = np.array(idx_tensor, dtype=np.float32)

        yaw_predicted = np.sum(
            yaw_predicted * idx_tensor_yaw, axis=1) * 3 - 180
        pitch_predicted = np.sum(pitch_predicted * idx_tensor, axis=1) * 3 - 99
        roll_predicted = np.sum(roll_predicted * idx_tensor, axis=1) * 3 - 99

        return np.array(yaw_predicted), np.array(pitch_predicted), np.array(roll_predicted)

    def whenet_draw(self, yaw, pitch, roll, tdx=None, tdy=None, size=200):
        """
        Plot lines based on yaw pitch roll values

        Args:
            yaw, pitch, roll: values of angles
            tdx, tdy: center of detected head area

        Returns:
            graph: locations of three lines
        """
        # taken from hopenet
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        tdx = tdx
        tdy = tdy

        # X-Axis pointing to right. drawn in red
        x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
        y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll)
                     * math.sin(pitch) * math.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
        y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch)
                     * math.sin(yaw) * math.sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (math.sin(yaw)) + tdx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy
        
        
        return {
            "yaw_x": x1,
            "yaw_y": y1, 
            "pitch_x": x2, 
            "pitch_y": y2, 
            "roll_x": x3, 
            "roll_y": y3
        }

''' Ends. '''
