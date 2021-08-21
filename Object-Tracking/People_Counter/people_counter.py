from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# argparse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# model classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow(args['prototxt'], args['model'])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# Video or Webcam
if not args.get('input', False):
    print("[INFO] starting Webcam...")
    vs = VideoStream(src=0).start()
    time.sleep(0.2)
else:
    print('[INFO] opening video file...')
    vs = cv2.VideoCapture(args['input'])
    
# save video
writer = None

# Shape of image
W, H = None, None

# load Tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableobjects = {}

# count
totalFrames = 0
totalDown = 0
totalUp = 0

# FPS estimator
fps = FPS().start()

# Loop Frames
while True:
    # read frames
    frame = vs.read()
    frame = frame[1] if args.get('input', False) else frame
    
    # Exceptions
    if args['input'] is not None and frame is None:
        break
    
    # image preprocessing
    frame = imutils.resize(frame, width=700)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        
    # save output
    if args['output'] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # 코덱
        writer = cv2.VideoWriter(args['output'], fourcc, 30, (W, H), True)
        
    # current status
    status = 'Waiting'
    rects = [] # bbox of every frame
    
    # N frame마다 detection 수행
    if totalFrames % args['skip_frames'] == 0:
        status = 'Detecting'
        trackers = []
        
        # frame to blob for detection model
        #blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        # detection
        net.setInput(blob)
        detections = net.forward()
        
        # loop detections
        for i in range(0, detections.shape[2]):
            
            confidence = detections[0, 0, i, 2] # confidence score
            if confidence > args['confidence']:
                
                # only person class
                idx = int(detections[0, 0, i, 1]) # index of class
                if idx != 0:
                   continue
                    
                # bbox of person clss
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype('int')
                
                # apply tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
        
    #detecting 안하고 Tracking만 할 때
    else:
        for tracker in trackers:
            status = 'Tracking'
            
            # update tracker
            tracker.update(rgb)
            pos = tracker.get_position() # dlib output 방식
            
            # unpack position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            
            # bbox of every frame
            rects.append((startX, startY, endX, endY))
        
        
    # draw line for count
    cv2.line(frame, (0, H//2), (W, H//2), (0, 255, 255), 2)
    
    # rects = detction을 통해 얻은 새로운 bbox와 tracker로 update된 bbox의 집합
    # 각 rects별 centroid 좌표를 갱신한다.
    print(rects)
    objects = ct.update(rects)
    
    # 갱신된 object들을 이용해 count        
    for (objectID, centroid) in objects.items():
        
        # 현재 track하고 있는 object의 기록
        to = trackableobjects.get(objectID, None)
        
        # 만약 track하고 있는 object의 기록이 없으면
        # 기록 추가
        if to is None:
            to = TrackableObject(objectID, centroid)
            
        # 만약 track하고 있는 object가 있었다면
        else:
            # 현재 centroid와 과거 centroids의 평균값을 빼서
            # 움직이고 있는 방향을 계산
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid) # centroid history 만들기
            
            # count된 이력이 있는지 확인
            if not to.counted:
        
                # 방향별로 Up 또는 Down count
                if (direction < 0) and (centroid[1] < (H//2)):
                    totalUp += 1
                    to.counted = True
                    
                elif (direction > 0) and (centroid[1] > (H//2)):
                    totalDown += 1
                    to.counted = True
        
        # 현재 track중인 object 명단에 update            
        trackableobjects[objectID] = to

        text = f'ID {objectID}'
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0),-1)
    
    # display current status    
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
    
    for i, (k, v) in enumerate(info):
        text = '{} : {}'.format(k, v)
        cv2.putText(frame, text, (10, H-((i*20) + 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    if writer is not None:
        writer.write(frame)
        
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    totalFrames += 1
    fps.update()
    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()
    
if not args.get('input', False):
    vs.stop()
    
else:
    vs.release()
    
cv2.destroyAllWindows()