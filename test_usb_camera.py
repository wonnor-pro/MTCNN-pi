import numpy as np
import cv2

cap = cv2.VideoCapture(0) # use camera 0

#brightness = +0.0
#contrast = +0.0
#white_balance_u = +2.0
#white_balance_v = +2.5
#iso_speed = 100
#framerate = 3
    
#cap.set(10, brightness)
#cap.set(11, contrast)
#white balance currently not supported?
#cap.set(CV_CAP_PROP_WHITE_BALANCE_U, white_balance_u)
#cap.set(CV_CAP_PROP_WHITE_BALANCE_V, white_balance_v)
#cap.set(CV_CAP_PROP_ISO_SPEED, iso_speed)
#cap.set(5,framerate)

print(cv2.VideoCapture.isOpened(cap))

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
