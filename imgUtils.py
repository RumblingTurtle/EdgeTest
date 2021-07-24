import cv2

def getBBoxCentroid(bbox):
    """
    Takes a bbox array [x1,y1,x2,y2] and returns coordinates of the bbox center
    :param bbox: bounding box coordinates
    :return: Tuple containing bounding box center coordinates
    """
    return bbox[0]+(bbox[2]-bbox[0])/2,bbox[1]+(bbox[3]-bbox[1])/2


def plotPaths(objectTracker, frame):
    """
    Takes a frame and tracking object and draws individual tracker paths on it
    :param objectTracker: Tracking object 
    :param frame: Frame used for drawing
    """
    
    x_shape, y_shape = frame.shape[1], frame.shape[0] #shape for frame normalized coordinates scaling
    for track in objectTracker.tracks:
        for i in range(len(track.bboxes)-1):
            x1,y1 = getBBoxCentroid(track.bboxes[i])
            x2,y2 = getBBoxCentroid(track.bboxes[i+1])

            cv2.line(frame,(int(x1*x_shape),int(y1*y_shape)),(int(x2*x_shape),int(y2*y_shape)),(0,255,0),3)

def plotBoxes(objectTracker, frame):
    """
    Takes a frame and tracking object and draws bounding boxes of rach track
    :param objectTracker: Tracking object 
    :param frame: Frame used for drawing
    """
    bboxes = [track.bboxes[-1] for track in objectTracker.tracks] #take the last bounding box for each track
    x_shape, y_shape = frame.shape[1], frame.shape[0] #shape for frame normalized coordinates scaling
    for bbox in bboxes:
        if bbox[4] >= 0.2:
            x1, y1, x2, y2 = int(bbox[0]*x_shape), int(bbox[1]*y_shape), int(bbox[2]*x_shape), int(bbox[3]*y_shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)



def IoU(boxA,boxB):
    """
    Calculates intersection over union for two bounding boxes (bbox arrays [x1,y1,x2,y2])
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea =  max(xB - xA + 1, 0) * max(yB - yA + 1,0)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
