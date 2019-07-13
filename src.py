import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
ap.add_argument("-s", "--savepath", required=True,
	help="path to directory which images will be saved")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(args['video'])

count = 0
print("scanning faces...")
while cap.isOpened():
    ret,frame = cap.read()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX < 0:
                startX = 0
            if startY < 0 :
                startY = 0
            if endX > w:
                endX = w-1
            if endY > h:
                endY = h-1
            # print(startX, startY, endX, endY)
            if 0 <= startX <= w and 0 <= endX <= w and 0 <= startY <= h and 0 <= endY <= h:
                cv2.imshow('face',frame[startY:endY, startX:endX])
                cv2.imwrite(args["savepath"]+"face%d.jpg" % count, frame[startY:endY, startX:endX])
    

    # cv2.imshow('window-name',frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()  #