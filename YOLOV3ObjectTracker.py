from imgUtils import IoU

class Track:
    def __init__(self,bbox,timeToLive=5,maxPointCount=20,assignmentThreshold=0.7):
        """
        Class representing tracked objects
        :param bbox: Starting bounding box
        :param timeToLive: Track's remaining iterations until detetion
        :param maxPointCount: Maximum amound of last M bounding boxes for a track
        :param assignmentThreshold: Lowest possible IoU value for assignment
        """
        self.maxTimeToLive = timeToLive
        self.timeToLive = timeToLive
        self.maxPointCount = maxPointCount
        self.bboxes = [bbox]
        self.assignmentThreshold = assignmentThreshold 

    def update(self,newBBoxes):
        """
        Updates current track given the array of the new bounding boxes
        :param newBBoxes: new set of bounding boxes
        """
        currentBBox = self.bboxes[-1]
        biggestIoU = 0
        candidateIdx = None
        
        for idx,bBox in enumerate(newBBoxes):
            iou = IoU(currentBBox,bBox)
            if iou > biggestIoU and iou>self.assignmentThreshold:
                biggestIoU = iou
                candidateIdx = idx

        
        isAlive = True
        #if no candidate has been found
        if candidateIdx == None:
            self.timeToLive-=1
            if self.timeToLive==0:
                isAlive = False
        else:
            self.timeToLive = self.maxTimeToLive 
            self.bboxes.append(newBBoxes[candidateIdx])

            #pop the first element of the path
            if len(self.bboxes)>self.maxPointCount:
                self.bboxes.pop(0)
        
        return isAlive, candidateIdx
            

class YOLOV3ObjectTracker:
    """
    Class that aggregates and updates current tracks
    """
    def __init__(self):
        self.tracks = []

    def updateTracks(self,bboxes):
        """
        Updates current tracks given new bounding boxes in the sequence
        :param bboxes: New set of detected bounding boxes
        """
        newTracks = []

        if len(self.tracks)==0: 
            #If there are no existing tracks
            #We assign every detection to a new track
            for bbox in bboxes:
                newTracks.append(Track(bbox))
        else:
            for track in self.tracks:
                isAlive,idx = track.update(bboxes)

                #If the update function of the track found it's candidate
                if idx!=None: 
                    bboxes.pop(idx)

                #Ignore dead tracks
                if isAlive:
                    newTracks.append(track)
            
            #Assign the bboxes that were not assigned to existing tracks as new
            for bbox in bboxes:
                newTracks.append(Track(bbox))

        self.tracks = newTracks