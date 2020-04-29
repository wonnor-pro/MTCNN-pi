import numpy as np
import matlab_wrapper
import cv2

def imResample(img, hs, ws):
    matlab = matlab_wrapper.MatlabSession()
    matlab.put('img', img)
    matlab.put('hs', hs)
    matlab.put('ws', ws)
    matlab.eval('script')
    im_data = matlab.get('im_data')
    return im_data
