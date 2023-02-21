import cv2


def set_params(min_area):
    params = cv2.SimpleBlobDetector_Params()

    #change threasholds
    params.minThreshold = 50
    params.maxThreshold = 2000

    #filter by area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 50000

    params.filterByCircularity = False
    params.filterByConvexity = False

    return params

def create_blob_detector(min_area):
    params = set_params(min_area)
    return cv2.SimpleBlobDetector_create(params)