import cv2
import dlib
import time

from copilot import LaneDetector, ModelType


# Set video resolution
WIDTH = 1280
HEIGHT = 720

# Set model
model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Set Cascade
carCascade = cv2.CascadeClassifier('haar-cascade/cars-2.xml')

# Initialize video
cap = cv2.VideoCapture("vid/1-2.mp4")

# Initialize lane detection model
laneDetector = LaneDetector(model_path, model_type, use_gpu)

cv2.namedWindow("CO-PILOT VIDEO", cv2.WINDOW_NORMAL)


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        carFrame = frame.copy()
        totalFrame = laneDetector.detect_lanes(frame)

        frameCounter = frameCounter + 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(carFrame)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(carFrame, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.16, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                windowHeight = frame.shape[0]
                y1 = round(windowHeight / 4)
                y2 = round(windowHeight * 5 / 6)

                windowWidth = frame.shape[1]
                x1 = round(windowWidth / 6)
                x2 = round(windowWidth * 5 / 6)

                if y1 < y < y2 and x1 < x < x2:
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchCarID = None

                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                                x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(carFrame, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(totalFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        cv2.putText(totalFrame, 'FPS: ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.imshow("CO-PILOT VIDEO", totalFrame)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()
