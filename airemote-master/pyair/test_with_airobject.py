import sys, os
import cv2 as cv

try:
    from airlib import AirLib

    air = AirLib('libairemote')
    air.use_remote('tcp://localhost:5530')

    ''' Create an AiRemote object and running throug a REMOTE! '''
    from googlenet.googlenet  import GoogleNet
    model = GoogleNet(air)

    #from picasso import Picasso
    #model = Picasso(air)

    ''' Only one picture is used '''
    image = cv.imread('rabit.jpg')

    ''' Run and print the result '''
    text = model.run(image)

except Exception as e:
    print(e)
 
''' Ends. '''
