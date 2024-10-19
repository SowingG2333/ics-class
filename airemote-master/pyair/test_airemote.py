import sys, os
import cv2 as cv

def run(model, image, remote):
    from airlib import AirLib
    air = AirLib('libairemote')
    air.use_remote(remote)

    input = model.pre_process(image)
    output = air.inference_remote(input)
    model.post_process(output)

def googlenet():
    from googlenet.googlenet import GoogleNet
    return GoogleNet()
   
def picasso():
    from picasso.picasso import Picasso
    return Picasso()

try:

    image = cv.imread('rabit.jpg')
    model = googlenet()
    remote = 'dummy://./googlenet/googlenet-dummy.bin'
    #remote = 'dummy://./picasso/picasso-dummy.bin'
    #remote='tcp://localhost:5530'

    run(model, image, remote)

except Exception as e:
    print(e)

''' Ends. '''
