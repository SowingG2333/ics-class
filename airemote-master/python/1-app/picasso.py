#!/usr/bin/env python3.9
"""Simplest app for GoogleNet """
'''
 USAGE:
    python {} <remote_string> <image_path>

    The remote_string goes like ``192.168.1.123'' or ``tcp://1.2.3.4:5678'' or 
    ``file:///home/HwHiAiUser/models/googlenet/googlenet.om''
'''
import os, sys
import cv2 as cv
import inferemote.airlab as lab

try:
  ''' Create an AiRemote object with a REMOTE string'''
  air = lab.Load(name='picasso', remote=sys.argv[1], verbose=True)
  ''' Take one picture as input '''
  image = cv.imread(sys.argv[2])
  outfile = os.path.join('.', 'result-' \
            + os.path.basename(sys.argv[2]))
  ''' Run inference and get the result in image'''
  result = air.inference(image)
  ''' Write the result in image '''
  cv.imwrite(outfile, result)

except:
  print("\n Usage: python {} <remote> <image>\n".format(sys.argv[0]))
''' Ends. '''
