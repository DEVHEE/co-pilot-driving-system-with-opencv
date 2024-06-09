import cv2

from copilot import LaneDetector, ModelType


model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

image_path = "img/input.png"

# Initialize lane detection model
laneDetector = LaneDetector(model_path, model_type, use_gpu)

# Read RGB images
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Detect the lanes
output_img = laneDetector.detect_lanes(img)

# Draw estimated depth
cv2.namedWindow("CO-PILOT IMAGE", cv2.WINDOW_NORMAL)
cv2.imshow("CO-PILOT IMAGE", output_img)
cv2.waitKey(0)

cv2.imwrite("img/output.png",output_img)
