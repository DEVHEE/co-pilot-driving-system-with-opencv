import cv2
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum

from copilot.model import parsingNet


lane_colors = [(0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 255, 0)]

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
    TUSIMPLE = 0
    CULANE = 1


class ModelConfig():

    def __init__(self, model_type):

        if model_type == ModelType.TUSIMPLE:
            self.init_tusimple_config()
        else:
            self.init_culane_config()

    def init_tusimple_config(self):
        self.img_w = 1280
        self.img_h = 720
        self.row_anchor = tusimple_row_anchor
        self.griding_num = 100
        self.cls_num_per_lane = 56

    def init_culane_config(self):
        self.img_w = 1640
        self.img_h = 590
        self.row_anchor = culane_row_anchor
        self.griding_num = 200
        self.cls_num_per_lane = 18


class LaneDetector():

    def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

        self.use_gpu = use_gpu

        # Load model configuration based on the model type
        self.cfg = ModelConfig(model_type)

        # Initialize model
        self.model = self.initialize_model(model_path, self.cfg, use_gpu)

        # Initialize image transformation
        self.img_transform = self.initialize_image_transform()

    @staticmethod
    def initialize_model(model_path, cfg, use_gpu):

        # Load the model architecture
        net = parsingNet(pretrained=False, backbone='18', cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, 4),
                         use_aux=False)  # do not need auxiliary segmentation in testing

        # Load the weights from the downloaded model
        if use_gpu:
            if torch.backends.mps.is_built():
                net = net.to("mps")
                state_dict = torch.load(model_path, map_location='mps')['model']  # Apple GPU
                print("Using Apple GPU")
            else:
                net = net.cuda()
                state_dict = torch.load(model_path, map_location='cuda')['model']  # CUDA
                print("Using CUDA")
        else:
            state_dict = torch.load(model_path, map_location='cpu')['model']  # CPU
            print("Using CPU")

        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        # Load the weights into the model
        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()

        return net

    @staticmethod
    def initialize_image_transform():
        # Create transform operation to resize and normalize the input images
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return img_transforms

    def detect_lanes(self, image, draw_points=True):

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)

        # Process output data
        self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

        # Draw depth image
        visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

        return visualization_img

    def prepare_input(self, img):
        # Transform the image for inference
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        input_img = self.img_transform(img_pil)
        input_tensor = input_img[None, ...]

        if self.use_gpu:
            if not torch.backends.mps.is_built():
                input_tensor = input_tensor.cuda()

        return input_tensor

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)

        return output

    @staticmethod
    def process_output(output, cfg):
        # Parse the output of the model
        processed_output = output[0].data.cpu().numpy()
        processed_output = processed_output[:, ::-1, :]
        prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        processed_output = np.argmax(processed_output, axis=0)
        loc[processed_output == cfg.griding_num] = 0
        processed_output = loc

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        lanes_points = []
        lanes_detected = []

        max_lanes = processed_output.shape[1]
        for lane_num in range(max_lanes):
            lane_points = []
            # Check if there are any points detected in the lane
            if np.sum(processed_output[:, lane_num] != 0) > 2:

                lanes_detected.append(True)

                # Process each of the points for each lane
                for point_num in range(processed_output.shape[0]):
                    if processed_output[point_num, lane_num] > 0:
                        lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1,
                                      int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane - 1 - point_num] / 288)) - 1]
                        lane_points.append(lane_point)
            else:
                lanes_detected.append(False)

            lanes_points.append(lane_points)
        # return np.array(lanes_points), np.array(lanes_detected)
        return lanes_points, lanes_detected

    @staticmethod
    def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
        # Write the detected line points in the image
        visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_AREA)

        if draw_points:
            for lane_num, lane_points in enumerate(lanes_points):
                for lane_point in lane_points:
                    cv2.circle(visualization_img, (lane_point[0], lane_point[1]), 3, lane_colors[lane_num], -1)

        # Draw a mask for the current lane
        if lanes_detected[1] and lanes_detected[2]:
            lane_segment_img = visualization_img.copy()

            cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))],
                         color=(0, 255, 0))
            if lanes_points[0]:
                cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[0], np.flipud(lanes_points[1])))],
                             color=(0, 0, 255))
            if lanes_points[3]:
                cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[2], np.flipud(lanes_points[3])))],
                             color=(0, 0, 255))
            visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

            # center middle white line
            windowWidth = visualization_img.shape[1]
            windowHeight = visualization_img.shape[0]

            window_pt1_x = round(windowWidth / 2)
            window_pt1_y = round(windowHeight * 2 / 4)
            window_pt2_x = window_pt1_x
            window_pt2_y = windowHeight
            cv2.line(visualization_img, [window_pt1_x, window_pt1_y], [window_pt2_x, window_pt2_y], (255, 255, 255), 5)

            # vertical line road
            # left
            cv2.line(visualization_img, lanes_points[1][round(2 / 5 * len(lanes_points[1]))],
                     lanes_points[1][round(4 / 5 * len(lanes_points[1]))], (0, 255, 255), 4)
            # right
            cv2.line(visualization_img, lanes_points[2][round(2 / 5 * len(lanes_points[2]))],
                     lanes_points[2][round(4 / 5 * len(lanes_points[2]))], (0, 255, 255), 4)

            # horizontal road
            cv2.line(visualization_img, lanes_points[1][round(3 / 5 * len(lanes_points[1]))],
                     lanes_points[2][round(3 / 5 * len(lanes_points[2]))], (255, 0, 255), 2)

            # middle vertical line road
            # left
            lane_point_1_x = lanes_points[1][round(3 / 5 * len(lanes_points[1]))][0]
            lane_point_1_y = lanes_points[1][round(3 / 5 * len(lanes_points[1]))][1]
            cv2.line(visualization_img, [lane_point_1_x, lane_point_1_y - 15], [lane_point_1_x, lane_point_1_y + 15],
                     (255, 0, 0), 3)
            # right
            lane_point_2_x = lanes_points[2][round(3 / 5 * len(lanes_points[2]))][0]
            lane_point_2_y = lanes_points[2][round(3 / 5 * len(lanes_points[2]))][1]
            cv2.line(visualization_img, [lane_point_2_x, lane_point_2_y - 15], [lane_point_2_x, lane_point_2_y + 15],
                     (255, 0, 0), 3)

            # center var
            lane_point_12_x = round((lane_point_1_x + lane_point_2_x) / 2)
            lane_point_12_y = round((lane_point_1_y + lane_point_2_y) / 2)

            # direction line
            cv2.line(visualization_img, [lane_point_12_x, lane_point_12_y], [window_pt2_x, window_pt2_y], (0, 0, 255),
                     2)

            # center line
            cv2.line(visualization_img, [lane_point_12_x, lane_point_12_y - 15],
                     [lane_point_12_x, lane_point_12_y + 15], (0, 255, 0), 3)

            if window_pt1_x - lane_point_12_x > 15:
                if window_pt1_x - lane_point_12_x > 65:
                    text = "CHANGING LANE"
                    cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))],
                                 color=(0, 255, 255))
                    visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
                else:
                    text = "TURN LEFT"
            elif window_pt1_x - lane_point_12_x < -15:
                if window_pt1_x - lane_point_12_x < -65:
                    text = "CHANGING LANE"
                    cv2.fillPoly(lane_segment_img, pts=[np.vstack((lanes_points[1], np.flipud(lanes_points[2])))],
                                 color=(0, 255, 255))
                    visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
                else:
                    text = "TURN RIGHT"
            else:
                text = " STRAIGHT"

            cv2.putText(visualization_img, text, (window_pt1_x - 140, round(windowHeight - 70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)
            cv2.putText(visualization_img, text, (window_pt1_x - 140, round(windowHeight - 70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        return visualization_img
