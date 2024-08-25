#This is the local .py file for running the SuperResolution code on your local machine natively

import cv2
from cv2 import dnn_superres

# Create an SR object and read image
sr = dnn_superres.DnnSuperResImpl_create()
image = cv2.imread("Remember Reach.jpg")

# Read the desired model
path = "FSRCNN_x2.pb"   #Make sure the SuperRes model is in the same directory as the .py file, or else change the path here to include the full directory
sr.readModel(path)

# Optionally, set CUDA backend and target to enable GPU inference if supported
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("fsrcnn", 2)

# Upscale the image and save result
result = sr.upsample(image)
cv2.imwrite("SuperResOutput.png", result)