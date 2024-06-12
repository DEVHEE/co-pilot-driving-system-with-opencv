import cv2
from copilot import LaneDetector, ModelType


model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize lane detection model
laneDetector = LaneDetector(model_path, model_type, use_gpu)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("CO-PILOT WEBCAM", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    # Detect the lanes
    output_img = laneDetector.detect_lanes(frame)

    cv2.imshow("CO-PILOT WEBCAM", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
