import cv2
from datetime import datetime

camera = cv2.VideoCapture(0)

for i in range(10):
    now = datetime.now()
    return_value, image = camera.read()
    cv2.imwrite('opencv'+str(i)+str(now)+'.png', image)
del(camera)