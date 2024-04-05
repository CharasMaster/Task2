import cv2 as cv
import numpy as np
from utils import *

def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    filterResponses = []
    for filter in filterBank:
        # Apply the filter to each channel of the image
        responses = [cv.filter2D(I[:, :, i], -1, filter) for i in range(3)]
        filterResponses.append(responses)

    return filterResponses

