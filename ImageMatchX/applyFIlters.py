import cv2 as cv
import numpy as np
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses

# Load the input image
image_path = 'place.jpg'
I = cv.imread(image_path)

# Create the filter bank
filterBank = create_filterbank()

# Extract filter responses
filterResponses = extract_filter_responses(I, filterBank)

# Display the filtered images for each channel
for i in range(3):  # Loop over each color channel
    channel_responses = [response[i] for response in filterResponses]  # Get responses for this channel
    channel_image = np.sum(np.stack(channel_responses), axis=0)  # Sum the responses for this channel
    cv.imshow(f'Filtered Channel {i}', channel_image.astype(np.uint8))

cv.waitKey(0)
cv.destroyAllWindows()
