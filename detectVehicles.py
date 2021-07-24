from YOLOV3ObjectTracker import YOLOV3ObjectTracker
from YOLOV3Detector import YOLOV3Detector
from imgUtils import plotBoxes,plotPaths
import numpy as np
import cv2
from time import time
import cv2
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='YOLOV3 based object detection and tracking script')

    parser.add_argument('inputPath', type=str,
                    help='Input video file path')
    parser.add_argument('outputPath', type=str,
                    help='output video file path')

    args = parser.parse_args()

    if args.outputPath==None or args.inputPath==None:
        raise Exception("Both input and output files have to be provided")

    outFile = args.outputPath
    inFile = args.inputPath

    detector = YOLOV3Detector()
    tracker = YOLOV3ObjectTracker()

    player = cv2.VideoCapture(inFile)
    assert player.isOpened()

    x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(outFile, four_cc, 20, (x_shape, y_shape))

    while True:
        ret, frame = player.read()
        assert ret

        start_time = time()
        blurred = cv2.GaussianBlur(frame,(3,3),0)
        result = detector.detect(blurred)
        tracks = tracker.updateTracks(result)
        end_time = time()


        plotBoxes(tracker, frame)
        plotPaths(tracker, frame)
        out.write(frame)
        
        
        fps = 1/np.round(end_time - start_time, 3)
        print(f"Frames Per Second : {fps}")
        cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow('frame',frame)
        #pressing Q stops the script
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("Done")