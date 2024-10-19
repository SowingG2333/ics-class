#!/usr/bin/env python3.9
import sys, cv2
import inferemote.airlab as lab
try:
  ''' Create an AiRemote object with a REMOTE string'''
  gnet = lab.Load(name='googlenet', remote=sys.argv[1])
  ''' Take one picture as input '''
  image = cv2.imread(sys.argv[2])
  ''' Run inference and print the result '''
  text = gnet.inference(image)
  print(f"\n Label predicted: ``{text}''\n")
except:
  print("\n Usage: python {} <remote> <image>\n".format(sys.argv[0]))

''' Ends. '''
