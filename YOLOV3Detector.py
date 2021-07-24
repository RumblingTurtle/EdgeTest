import torch

class YOLOV3Detector:
    #COCO classes
    CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

    #Detectable class list
    CLASS_WHITELIST = ['car', 'motorcycle', 'bus', 'train', 'truck']

    def __init__(self):  
        self.model = torch.hub.load('ultralytics/yolov3', 'yolov3_tiny', pretrained=True)
        self.model.to('cpu') # as stated in the Bonus tasks
        self.model.conf = 0.25 # confidence threshold

    def detect(self,frame):
        """
        Takes a single frame as input, and scores the frame using yolov3 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Bounding box parameters of detected objects in the frame.
        """
        results = self.model(frame, size=512)
        labels, bboxes = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

        outBBoxes = []
        #Filtering by label
        for label,bbox in zip(labels,bboxes):
            if self.CLASSES[int(label)] in self.CLASS_WHITELIST:
                outBBoxes.append(bbox)
        return outBBoxes
