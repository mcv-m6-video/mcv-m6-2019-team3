from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import motmetrics as mm
from tqdm import tqdm
import cv2
from object_tracking.tracking import rgb_histogram
from utils.detection import Detection

class CentroidTracker():
	
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
       
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
    def register(self, centroid, width, height):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = [centroid, width, height]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
    def update(self, rects):
        width = 0
        height = 0
		# check to see if the list of input bounding box rectangles
		# is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
            	self.disappeared[objectID] += 1

            	# if we have reached a maximum number of consecutive
            	# frames where a given object has been marked as
            	# missing, deregister it
            	if self.disappeared[objectID] > self.maxDisappeared:
            		self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
        	# use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            width = endX-startX
            height = endY-startY
            inputCentroids[i] = (cX, cY)


        # if we are currently not tracking any objects take the input
		# centroids and register each of them
        if len(self.objects) == 0:
        	for i in range(0, len(inputCentroids)):
        		self.register(inputCentroids[i], width, height)
        # otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            # print(self.objects.values()[0,:])
            # print(list(self.objects.values())[:,0])
            objectCentroids = [x[0] for x in list(self.objects.values())]
            # print(objectCentroids)

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()            
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
            	# if we have already examined either the row or
            	# column value before, ignore it
            	# val
            	if row in usedRows or col in usedCols:
            		continue            
            	# otherwise, grab the object ID for the current row,
            	# set its new centroid, and reset the disappeared
            	# counter
            	objectID = objectIDs[row]
            	self.objects[objectID] = [inputCentroids[col], width, height]
            	self.disappeared[objectID] = 0          
            	# indicate that we have examined each of the row and
            	# column indexes, respectively
            	usedRows.add(row)
            	usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1                 
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
            	for col in unusedCols:
            		self.register(inputCentroids[col], width, height)

        # return the set of trackable objects
        return self.objects


def track_objects(ct, video_path, detections_list, gt_list, display = False, export_frames = False, idf1 = True):

    colors = np.random.rand(500, 3)  # used only for display
    tracks = []
    max_track = 0
    new_detections = []

    if idf1:
        acc = mm.MOTAccumulator(auto_id=True)

    capture = cv2.VideoCapture(video_path)
    n_frame = 0
    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break
        frame_tracks = {}

        detections_on_frame = [x for x in detections_list if x.frame == n_frame]
        # print("Detections on fr: {}".format(detections_on_frame))
        gt_on_frame = [x for x in gt_list if x.frame == n_frame]
        rects = [[x.xtl, x.ytl, x.xtl+x.width, x.ytl+x.height] for x in detections_on_frame]
        # print(rects)
        objects = ct.update(rects)
        if display:
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(image, text, (centroid[0][0] - 10, centroid[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image, (centroid[0][0], centroid[0][1]), 4, (0, 255, 0), -1)

            # show the output frame
            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF
            # cv2.waitKey()

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if display and n_frame%2==0 and n_frame < 200:
        #     visualize_tracks(image, frame_tracks, colors, display=display)

        # if export_frames:
        #     visualize_tracks_opencv(image, frame_tracks, colors, export_frames=export_frames,
        #                      export_path="output_frames/tracking/frame_{:04d}.png".format(n_frame))

        # IDF1 computing
        detec_bboxes = []
        detec_ids = []
        # for key, value in frame_tracks.items():
        for (objectID, centroid) in objects.items():
            detec_ids.append(objectID)
            bbox = [centroid[0][0], centroid[0][1], centroid[1]+centroid[0][0], centroid[2]+centroid[0][1]]
            # print(bbox)
            conf = 1
            detec_bboxes.append(bbox)
            new_detections.append(Detection(n_frame, 'car', bbox[0], bbox[1], bbox[2] - bbox[0],
                                            bbox[3] - bbox[1], conf,
                                            track_id=objectID, histogram=rgb_histogram(image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :])))


        gt_bboxes = []
        gt_ids = []
        for gt in gt_on_frame:
            gt_bboxes.append(gt.bbox)
            gt_ids.append(gt.track_id)

        mm_gt_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in gt_bboxes]
        mm_detec_bboxes = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2, bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in detec_bboxes]

        distances_gt_det = mm.distances.iou_matrix(mm_gt_bboxes, mm_detec_bboxes, max_iou=1.)
        if idf1:
            acc.update(gt_ids, detec_ids, distances_gt_det)

        pbar.update(1)
        n_frame += 1

    pbar.close()
    capture.release()
    cv2.destroyAllWindows()

    if idf1:
        print(acc.mot_events)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        print(summary)
        with open("results/metrics.txt", "a") as f:
            f.write(summary.to_string() + "\n\n")
    return new_detections
