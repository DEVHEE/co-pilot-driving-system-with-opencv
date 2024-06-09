import cv2

from copilot import LaneDetector, ModelType


model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize video
cap = cv2.VideoCapture("vid/2-2.mp4")

# Initialize lane detection model
laneDetector = LaneDetector(model_path, model_type, use_gpu)

cv2.namedWindow("CO-PILOT VIDEO", cv2.WINDOW_NORMAL)

while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()

    except:
        continue

    if ret:
        # Detect the lanes
        output_img = laneDetector.detect_lanes(frame)

        cv2.imshow("CO-PILOT VIDEO", output_img)

    else:
        break

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
