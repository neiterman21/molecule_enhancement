import cv2


def set_params(min_area):
    params = cv2.SimpleBlobDetector_Params()

    #change threasholds
    params.minThreshold = 50
    params.maxThreshold = 100

    #filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 80000
    params.minDistBetweenBlobs = 100

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByColor = False

    return params

def create_blob_detector(min_area):
    params = set_params(min_area)
    return cv2.SimpleBlobDetector_create(params)

def get_blobs(frame,min_area):
    d1 = cv2.SimpleBlobDetector_create(set_params(min_area))
    #d2 = cv2.SimpleBlobDetector_create(set_params(min_area*2))
    #o2 = d2.detect(frame)
    return d1.detect(frame)
