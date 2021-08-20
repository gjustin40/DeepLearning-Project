from imutils.video import VideoStream
from centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import time
import cv2

prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
confidence = 0.8

ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video streams...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    try:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
        
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > confidence:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))
                
                (startX, startY, endX, endY) = box.astype('int')
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
        objects = ct.update(rects)
        
        for (objectID, centroid) in objects.items():
            text = f'ID {objectID}'
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] -10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
            
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
    except:
        print('No Objects')
    
cv2.destropyAllWindows()
vs.stop()   