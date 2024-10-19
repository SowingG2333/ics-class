#!/usr/bin/env python3.9
"""
  Simplest app for Portrait 

"""
import os, sys, cv2
import inferemote.airlab as lab

''' Load an object from airlab '''
#air = lab.Load(name='portrait', remote=sys.argv[1], verbose=True)
air = lab.Load(name='portrait')

''' Only one picture is used '''
image = cv2.imread(sys.argv[2])
bg_image = cv2.imread(sys.argv[3])

''' Run and print the result '''
mask = air.inference(image)
new_img = air.make_result(image, mask, bg_image)

''' Write result picture '''
outfile = os.path.join('.', 'result-' + os.path.basename(sys.argv[2]))
if cv2.imwrite(outfile, new_img):
    print(f"Result image written in ``{outfile}''.")

''' Ends. '''
