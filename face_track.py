import cv2
import numpy as np
from motpy import MultiObjectTracker, Detection
from motpy.core import setup_logger
from motpy.testing_viz import draw_track

webcam_src = 0  #webcam_src = 1 if using external USB camera 
logger = setup_logger(__name__, 'DEBUG', is_main=True)
font = cv2.FONT_HERSHEY_SIMPLEX

#Initialising face coordinates
x_old = 0
y_old = 0

def detectFaceDNN(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300,300), [104, 117, 123], False, False,
        )
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    out_detections = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            out_detections.append(Detection(box=[x1,y1,x2,y2], score=confidence))
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, bboxes, out_detections

def StorePreviousValues(x, y):
    x_old = x
    y_old = y
    return x_old, y_old
    
def ExtractBoxValues(tracks):
    x_new = 0
    y_new = 0
    if len(tracks)!= 0 :
        x_new = tracks[0][1][0]
        y_new = tracks[0][1][1]
    return x_new, y_new

#Including face detection OpenCV neural network model file
modelFile = "caffe_model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "caffe_model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

#face tracking
model_spec = {'order_pos': 1, 'dim_pos': 2,
                  'order_size': 0, 'dim_size': 2,
                  'q_var_pos': 5000., 'r_var_pos': 0.1}
dt = 1 / 30.0  # assume 30 fps
tracker = MultiObjectTracker(dt=dt, model_spec=model_spec)

# perform face detection on webcam
video_capture = cv2.VideoCapture(webcam_src)
while True:
    ret, frame = video_capture.read()
    if frame is None:
        break
    frame = cv2.resize(frame, dsize=None, fx=1.98, fy=1.37)
    outOpencvDnn , bboxes , detections= detectFaceDNN(net, frame)
    logger.debug(f'detections: {bboxes}')
    
    #tracking take place with the help of motpy library
    tracker.step(detections)
    tracks = tracker.active_tracks(min_steps_alive=3)
    logger.debug(f'tracks: {tracks}')

    # people counting algorithm (to be completed)
    print(outOpencvDnn.shape)
    x_new, y_new = ExtractBoxValues(tracks)
    print("x_old = " + str(x_old))
    print("y_old = " + str(y_old))
    print("x_new = " + str(x_new))
    print("y_new = " + str(y_new))
    if(x_old > x_new):
        print("Left")
    x_old, y_old = StorePreviousValues(x_new, y_new)
    
    
    for track in tracks:
        draw_track(outOpencvDnn, track)
    total_persons = len(bboxes)
    cv2.putText(outOpencvDnn,
                'Persons in frame = '+str(total_persons),
                (200,25),
                font,
                1,(255,0,0),2)
    cv2.imshow('Video', outOpencvDnn)
    if cv2.waitKey(1) & 0xFF == ord('p'): # press 'p' to terminate program
        break
video_capture.release()
cv2.destroyAllWindows()
    
