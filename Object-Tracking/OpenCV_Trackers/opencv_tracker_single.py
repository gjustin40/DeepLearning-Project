from imutils.video import VideoStream, FPS
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create(),
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

tracker = OPENCV_OBJECT_TRACKERS['kcf']

initBB = None
fps = None

if not args.get('video', False):
    print("[INFO} starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1)
    
else:
    vs = cv2.VideoCapture(args['video'])
    

while True:
    frame = vs.read()
    if args.get('video', False):
        frame = frame[1]

    if frame is None:
        break
    
    frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]
    
    if initBB is not None:
        success, box = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        fps.update()
        fps.stop()
        
        info = [
            ('Tracker', args['tracker']),
            ('Success', 'Yes' if success else 'No'),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        
        for (i, (k, v)) in enumerate(info):
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, H-((i*20) + 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        
        tracker.init(frame, initBB)
        fps = FPS().start()
        
    elif key == ord('q'):
        break
    
if not args.get('video', False):
    vs.stop()
    
else:
    vs.release()
    
cv2.destroyAllWindows()
        
    
    