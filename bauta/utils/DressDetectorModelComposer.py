import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, random
from torch.autograd import Variable
import sys
import cv2
import traceback
import numpy as np
from bauta.BoundingBox import BoundingBox
from bauta.utils.ImageUtils import ImageUtils
import math

class DressDetectorModelComposer():
    """Generates randomly models wearing a dress."""

    def __init__(self, is_train, data_path, color_distortion=True, seed=None):
        random.seed(seed)
        self.image_utils = ImageUtils()
        self.data_path = data_path
        self.armless_model_image_path = os.path.join(self.data_path, 'images-model/base_model_beig_armless.png')
        self.armless_legless_model_image_path = os.path.join(self.data_path, 'images-model/base_model_beig_armless_legless.png')
        self.right_leg_image_path = os.path.join(self.data_path, 'images-model/right_leg.png')
        self.heads_path = os.path.join(self.data_path, 'images-model/heads')
        self.upperarm_image_path = os.path.join(self.data_path, 'images-model/upperarm.png')
        self.forearm_image_path = os.path.join(self.data_path, 'images-model/forearm.png')
        self.forearm_image = cv2.imread(self.forearm_image_path, cv2.IMREAD_UNCHANGED)
        self.upperarm_image = cv2.imread(self.upperarm_image_path, cv2.IMREAD_UNCHANGED)
        self.model_image = cv2.imread(self.armless_model_image_path, cv2.IMREAD_UNCHANGED)
        self.model_image_legless = cv2.imread(self.armless_legless_model_image_path, cv2.IMREAD_UNCHANGED)
        self.model_right_leg_image = cv2.imread(self.right_leg_image_path, cv2.IMREAD_UNCHANGED)
        self.x_center_right_upper_arm = 25
        self.y_center_right_upper_arm = 40
        self.x_center_right_upper_arm_end = 45
        self.y_center_right_upper_arm_end = 265
        self.x_center_left_upper_arm = 105 - self.x_center_right_upper_arm
        self.y_center_left_upper_arm = self.y_center_right_upper_arm
        self.x_center_left_upper_arm_end = 136 - 45
        self.y_center_left_upper_arm_end = 265
        self.x_center_left_upper_arm = 105 - self.x_center_right_upper_arm
        self.y_center_left_upper_arm = self.y_center_right_upper_arm
        self.arm_spread_width_offset = 200
        self.color_distortion = color_distortion

    def randomColor(self, garment_image):
        garment_image_bgr = garment_image[:,:,0:3]
        garment_image_hsv = cv2.cvtColor(garment_image_bgr,cv2.COLOR_BGR2HSV)
        hue = random.uniform(0, 178)
        garment_image_hsv[:,:,0] = hue
        saturation_gain = random.random() + 1.0
        garment_image_hsv[:,:,1] = garment_image_hsv[:,:,1] * saturation_gain
        garment_image_hsv[:,:,1] = ((garment_image_hsv[:,:,1] > 255) * 255) + \
            ((garment_image_hsv[:,:,1] <= 255) * garment_image_hsv[:,:,1])
        value_gain = random.random() + 1.0
        garment_image_hsv[:,:,2] = garment_image_hsv[:,:,2] * value_gain
        garment_image_hsv[:,:,2] = ((garment_image_hsv[:,:,2] > 255) * 255) + \
            ((garment_image_hsv[:,:,2] <= 255) * garment_image_hsv[:,:,2])
        garment_image_bgr = cv2.cvtColor(garment_image_hsv,cv2.COLOR_HSV2BGR)
        for color_channel in range(0, 3):
            garment_image[:,:,color_channel] = garment_image_bgr[:,:,color_channel]
        return garment_image, hue

    def leftAndRightCoordinatesForNeckline(self, garment_image):
        garment_image_height, garment_image_width, garment_image_colors = garment_image.shape
        garment_image_color_first = np.transpose(garment_image, axes=(2,0,1))
        garment_alpha_channel = garment_image_color_first[3]
        garment_alpha_channel_grayscale = garment_alpha_channel.reshape((garment_image_height, garment_image_width))
        garment_alpha_channel_grayscale = np.array(garment_alpha_channel_grayscale, np.uint8)
        # take the upper part of the dress (neckline)
        for delta in  np.arange(0.01, 1.0, 0.02):
            upper_part = garment_alpha_channel_grayscale[0:int(garment_image_height * delta), 0:garment_image_width]
            # find the bounding box using basic segmentation
            bounging_box_left  = None
            bounging_box_right = None
            bounding_boxes, _ = BoundingBox.fromOpenCVConnectedComponentsImage(upper_part)
            for bounding_box in bounding_boxes: 
                if bounding_box._right < garment_image_width / 2:
                    bounging_box_left = bounding_box
                if bounding_box._left  > garment_image_width / 2:
                    bounging_box_right = bounding_box
            if (bounging_box_left is not None) and (bounging_box_right is not None):
                return bounging_box_left.left, bounging_box_right._right
        return int(garment_image_width * 0.1), int(garment_image_width * 0.9)

    def getAverageHue(self, image):
        image_bgr = image[:,:,0:3]
        image_hsv = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
        average_hue = int(np.average(image_hsv[:,:,0]))
        return average_hue

    def upperArm(self, composed_image, upper_arm_angle_distortion, upperarm_image,\
        x_center_upper_arm, y_center_upper_arm, x_offset_model, x_offset):
        upper_arm_rotation_matrix = cv2.getRotationMatrix2D( \
            (x_center_upper_arm + self.arm_spread_width_offset, \
            y_center_upper_arm + self.arm_spread_width_offset), \
             upper_arm_angle_distortion, 1)
        upperarm_image = cv2.copyMakeBorder(upperarm_image, \
            self.arm_spread_width_offset, self.arm_spread_width_offset, \
            self.arm_spread_width_offset, self.arm_spread_width_offset, \
            cv2.BORDER_CONSTANT, value=0)
        upperarm_image = cv2.warpAffine(upperarm_image, upper_arm_rotation_matrix, \
            (upperarm_image.shape[1], upperarm_image.shape[0]))
        return self.image_utils.pasteRGBAimageIntoRGBimage(upperarm_image, composed_image,\
            x_offset_model + x_offset - self.arm_spread_width_offset, 120 - self.arm_spread_width_offset, skip_mask=True)

    def upperArms(self, composed_image, x_offset_model, model_image_width):
        right_upper_arm_angle_distortion = random.uniform(0, 30)
        composed_image = self.upperArm(composed_image, right_upper_arm_angle_distortion, self.upperarm_image, \
            self.x_center_right_upper_arm, self.y_center_right_upper_arm, \
            x_offset_model, model_image_width - 120)
        left_upperarm_image = cv2.flip(self.upperarm_image, 1)
        left_upper_arm_angle_distortion = -random.uniform(0, 30)
        self.upperArm(composed_image, left_upper_arm_angle_distortion, \
            left_upperarm_image, self.x_center_left_upper_arm, \
            self.y_center_left_upper_arm, x_offset_model, 0)
        return right_upper_arm_angle_distortion, left_upper_arm_angle_distortion

    def forearm(self, composed_image_forearms, forearm_image, forearm_angle_distortion, \
        upper_arm_angle_distortion, x_center_upper_arm_end, y_center_upper_arm_end, x_center_upper_arm, y_center_upper_arm, x_offset_model, offset):
        x_center_upper_arm_end_vector = x_center_upper_arm_end - x_center_upper_arm
        y_center_upper_arm_end_vector = y_center_upper_arm_end - y_center_upper_arm
        s = math.sin((upper_arm_angle_distortion / 360) * 2.0 * math.pi)
        c = math.cos((upper_arm_angle_distortion / 360) * 2.0 * math.pi)
        x_center_upper_arm_end_rotation = int((x_center_upper_arm_end_vector * c) \
            - (y_center_upper_arm_end_vector * s))
        y_center_upper_arm_end_rotation = int((x_center_upper_arm_end_vector * s) \
            + (y_center_upper_arm_end_vector * c))
        forearm_image = cv2.copyMakeBorder(forearm_image, self.arm_spread_width_offset,\
            self.arm_spread_width_offset, self.arm_spread_width_offset, \
            self.arm_spread_width_offset, cv2.BORDER_CONSTANT, value=0)
        forearm_rotation_matrix = cv2.getRotationMatrix2D( \
            (100 + self.arm_spread_width_offset, 30 + self.arm_spread_width_offset), \
            forearm_angle_distortion, 1)
        forearm_image = cv2.warpAffine(forearm_image, forearm_rotation_matrix, \
            (forearm_image.shape[1], forearm_image.shape[0]))
        fake_background = self.image_utils.blankImage(forearm_image.shape[0], forearm_image.shape[1])
        self.image_utils.pasteRGBAimageIntoRGBimage(forearm_image, composed_image_forearms, \
            x_offset_model + offset - self.arm_spread_width_offset - x_center_upper_arm_end_rotation, \
            120 - self.arm_spread_width_offset + y_center_upper_arm_end_rotation, skip_mask=True)

    def forearms(self, right_upper_arm_angle_distortion, left_upper_arm_angle_distortion, x_offset_model, garment_image_width, model_image):
        model_image_height, model_image_width, model_image_colors = model_image.shape
        composed_image_forearms = np.zeros((model_image_height, \
            max(garment_image_width, model_image_width) + self.arm_spread_width_offset, model_image_colors), dtype=np.uint8)
        right_forearm_angle_distortion = random.uniform(-30, 30)
        self.forearm(composed_image_forearms, self.forearm_image, \
            right_forearm_angle_distortion, right_upper_arm_angle_distortion, \
            self.x_center_right_upper_arm_end, self.y_center_right_upper_arm_end, \
            self.x_center_right_upper_arm, self.y_center_right_upper_arm, \
            x_offset_model, model_image_width - 120)
        left_forearm_image = cv2.flip(self.forearm_image, 1)
        left_forearm_angle_distortion = random.uniform(-50, 50)
        self.forearm(composed_image_forearms, left_forearm_image, \
            left_forearm_angle_distortion, left_upper_arm_angle_distortion, \
            self.x_center_left_upper_arm_end, self.y_center_left_upper_arm_end, \
            self.x_center_left_upper_arm, self.y_center_left_upper_arm, x_offset_model, 0)
        composed_image_forearms[:,:,3] = (composed_image_forearms[:,:,0]+composed_image_forearms[:,:,1]+composed_image_forearms[:,:,2] > 0) * 255
        return composed_image_forearms

    def rightLeg(self, right_leg_image, x_offset_model, garment_image_width, model_image):
        model_image_height, model_image_width, model_image_colors = model_image.shape
        composed_image_right_leg = np.zeros((model_image_height, \
            max(garment_image_width, model_image_width) + self.arm_spread_width_offset, model_image_colors), dtype=np.uint8)
        x_distortion = int(random.uniform(-60, 60))
        y_distortion = int(random.uniform(-30, 30))
        self.image_utils.pasteRGBAimageIntoRGBimage(right_leg_image, composed_image_right_leg, \
            x_offset_model + 250 + x_distortion , \
            550 + y_distortion, skip_mask=True)
        composed_image_right_leg[:,:,3] = (composed_image_right_leg[:,:,0]+composed_image_right_leg[:,:,1]+composed_image_right_leg[:,:,2] > 0) * 255
        return composed_image_right_leg

    def mergeModelWithArms(self, only_dress_composition, composed_image, composed_image_forearms):
        for channel in range(0, 4):
            only_dress_composition[:,:, channel] = only_dress_composition[:,:, channel] * \
                ((composed_image_forearms[:,:,0] + composed_image_forearms[:,:,1] + composed_image_forearms[:,:,2]) == 0)
        self.image_utils.pasteRGBAimageIntoRGBimage(composed_image_forearms, composed_image, 0, 0, skip_mask=True)
        composed_image[:,:,3] = (composed_image[:,:,0]+composed_image[:,:,1]+composed_image[:,:,2] > 0) * 255

    def addModel(self, garment_image):
        model_image = None
        show_single_leg = bool(random.getrandbits(1))
        if show_single_leg:
            model_image =  self.model_image_legless
        else:
            model_image = self.model_image

        model_image_height, model_image_width, model_image_colors = model_image.shape

        neckline_left, neckline_right = self.leftAndRightCoordinatesForNeckline(garment_image)
        neckline_width = neckline_right - neckline_left + 1

        garment_image_height, garment_image_width, garment_image_colors = garment_image.shape
        composed_image = np.zeros((model_image_height, \
            max(garment_image_width, model_image_width) + self.arm_spread_width_offset, model_image_colors), dtype=np.uint8)

        composed_image_height, composed_image_width, composed_image_colors = composed_image.shape
        x_offset_model = int((composed_image_width  - model_image_width) / 2)

        composed_image  = self.image_utils.pasteRGBAimageIntoRGBimage(model_image, composed_image, x_offset_model, 0, skip_mask=True)
        right_upper_arm_angle_distortion, left_upper_arm_angle_distortion = self.upperArms(composed_image, x_offset_model, model_image_width)
        composed_image_forearms = self.forearms(right_upper_arm_angle_distortion, \
            left_upper_arm_angle_distortion, x_offset_model, garment_image_width, \
            model_image)

        composed_image_height, composed_image_width, composed_image_colors = composed_image.shape
        x_offset_garment = int((composed_image_width  - neckline_width) / 2) - neckline_left
        y_offset_garment = 90
        composed_image = self.image_utils.pasteRGBAimageIntoRGBimage(garment_image, composed_image, x_offset_garment, y_offset_garment, False, skip_mask=True)
        only_dress_composition = np.zeros((model_image_height, \
            max(garment_image_width, model_image_width) + \
            self.arm_spread_width_offset, model_image_colors), dtype=np.uint8)
        only_dress_composition = self.image_utils.pasteRGBAimageIntoRGBimage(garment_image, \
            only_dress_composition, x_offset_garment, y_offset_garment, True, skip_mask=True)
        if show_single_leg:
            composed_image_right_leg = self.rightLeg(self.model_right_leg_image, x_offset_model, garment_image_width, \
                model_image)
            self.mergeModelWithArms(only_dress_composition, composed_image, composed_image_right_leg)
        self.mergeModelWithArms(only_dress_composition, composed_image, composed_image_forearms)
        return composed_image, only_dress_composition

    def compose(self, garment_image):
        scale = 2.5# to be checked 0.70
        hue = self.getAverageHue(garment_image)
        garment_image_height, garment_image_width, garment_image_colors = garment_image.shape
        apply_random_colour = self.color_distortion and bool(random.getrandbits(1))
        if apply_random_colour:
            garment_image, hue = self.randomColor(garment_image)
        garment_image = cv2.resize(garment_image, (int(scale * garment_image_width), \
            int(scale * garment_image_height)), interpolation = cv2.INTER_CUBIC)
        composed_image = None
        only_dress_composition = None
        add_model = np.random.uniform(0, 1) < 0.95
        if add_model:
            composed_image, only_dress_composition = self.addModel(garment_image)

        else:
            composed_image = garment_image
            only_dress_composition = garment_image

        apply_flip = bool(random.getrandbits(1))
        if apply_flip:
            composed_image = cv2.flip(composed_image, 1)
            only_dress_composition = cv2.flip(only_dress_composition, 1)

        return composed_image, only_dress_composition, hue
