import cv2
import math
import numpy as np
import random

from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo
from bauta.Constants import constants

class ImageDistortions():
    def __init__(self):
        self.image_utils = ImageUtils()

    def _getScaleValue(self):
        probability = random.uniform(0, 1)
        if probability < 0.7:
            scale = random.uniform(0.8, 1.0)
        elif 0.8 <= probability < 0.95:
            scale = random.uniform(0.5, 0.8)
        else:
            scale = random.uniform(0.1, 0.5)
        return scale

    def getScaleParams(self, original_width, original_height):
        proportion_y = original_width / original_height
        max_scale_x = constants.input_width / original_width
        max_scale_y = constants.input_height / original_height
        scale_x = self._getScaleValue()
        scale_y = self._getScaleValue()
        if scale_x > max_scale_x:
            scale_x = scale_x * max_scale_x * proportion_y
            scale_y = scale_y * max_scale_y
        elif scale_y > max_scale_y:
            scale_y = scale_y * max_scale_y / proportion_y
            scale_x = scale_x * max_scale_x
        return scale_x, scale_y

    def getScaleMatrix(self, scale_x, scale_y):
        scale_matrix = [
            [scale_x,   0,       0],
            [0,         scale_y, 0],
            [0,         0,       1]
        ]
        return np.array(scale_matrix)

    def getScaledRotoTranslationMatrix(self, scale_x, scale_y, original_width, original_height):
        angle_probability = random.uniform(0, 1)
        if angle_probability < 0.80:
            angle = random.uniform(-15, 15)
        elif 0.80 <= angle_probability < 0.90:
            angle = random.uniform(-30, 30)
        elif 0.90 <= angle_probability < 0.98:
            angle = random.uniform(-90, 90)
        else:
            angle = random.uniform(-180, 180)
        angle_radiants = math.radians(angle)
        cos_angle = math.cos(angle_radiants)
        sin_angle = math.sin(angle_radiants)
        image_width = original_height * math.fabs(sin_angle)  + original_width * math.fabs(cos_angle)
        image_height = original_height * math.fabs(cos_angle) + original_width * math.fabs(sin_angle)

        # center the rotated image on the top left angle of the image
        percentage_out_of_image = 0.2
        base_x_translation = ( (1 - cos_angle ) * original_width / 2 ) - ( sin_angle       * original_height / 2 ) + ( image_width / 2 - original_width / 2 )
        base_y_translation = ( sin_angle       *  original_width / 2 ) + ( (1 - cos_angle) * original_height / 2 ) + ( image_height / 2 - original_height / 2 )
        random_x_translation = random.uniform(-percentage_out_of_image * image_width, constants.input_width / scale_x - ( (image_width * (1 - percentage_out_of_image))  ) )
        random_y_translation = random.uniform(-percentage_out_of_image * image_height, constants.input_height / scale_y - ( (image_height * (1 - percentage_out_of_image)) ) )
        rotation_matrix = np.array([
	    [cos_angle,    sin_angle],
	    [-sin_angle,   cos_angle]
	])
        rototranslation_matrix  = np.zeros((3, 3))
        rototranslation_matrix[0:2,0:2] = rotation_matrix
        rototranslation_matrix[:, 2:3] = [
	    [base_x_translation + random_x_translation],
	    [base_y_translation + random_y_translation],
	    [1]
	]
        return rototranslation_matrix

    def getPerspectiveMatrix(self):
        perspective_probability = random.uniform(0, 1)
        if perspective_probability < 0.7:
            perspective_x = random.uniform(-0.00001, 0.00001)
            perspective_y = random.uniform(-0.00001, 0.00001)
        elif 0.7 <= perspective_probability < 0.95:
            perspective_x = random.uniform(-0.0001, 0.0002)
            perspective_y = random.uniform(-0.0001, 0.0002)
        else:
            perspective_x = random.uniform(-0.0003, 0.0005)
            perspective_y = random.uniform(-0.0003, 0.0005)

        perspective_matrix = [
          [1, 			0, 		0],
          [0,                   1, 		0],
          [perspective_x,       perspective_y,	1]
        ]
        return np.matrix(perspective_matrix)

    def normalize_contrast(self, channel, contrast, lower_threshold, upper_threshold):
        mean_channel = np.mean(channel)
        if mean_channel * contrast >= upper_threshold:
            contrast = contrast * mean_channel / math.fabs(upper_threshold - mean_channel * contrast)
        elif mean_channel * contrast <= lower_threshold:
            contrast = contrast * mean_channel / math.fabs(mean_channel * contrast - lower_threshold)
        return contrast

    def getContrastParameter(self, image):
        contrast_probability = random.uniform(0, 1)
        if contrast_probability < 0.7:
            contrast = random.uniform(-0.1, 0.1)
        elif 0.7 <= contrast_probability < 0.95:
            contrast = random.uniform(-0.3, 0.3)
        else:
            contrast = random.uniform(-0.5, 0.5)
        # 1 is the unaltered, without contrast, image
        contrast = 1 + contrast
        contrast_channels = [self.normalize_contrast(channel, contrast, lower_threshold=5, upper_threshold=250) \
                                for channel in cv2.split(image)]
        return min(contrast_channels)

    def normalizeBrightness(self, channel, brightness, lower_threshold, upper_threshold):
        mean_channel = np.mean(channel)
        if mean_channel + brightness  >= upper_threshold:
            brightness = math.fabs( (upper_threshold - mean_channel) / upper_threshold) * brightness
        elif mean_channel + brightness <= lower_threshold:
            brightness = math.fabs( (mean_channel - lower_threshold) / lower_threshold) * brightness
        return int(brightness)

    def getBrightnessParameter(self, image):
        brightness_probability = random.uniform(0, 1)
        if brightness_probability < 0.7:
            brightness = random.uniform(-10, 10)
        elif 0.7 <= brightness_probability < 0.85:
            brightness = random.uniform(-30, 30)
        elif 0.85 <= brightness_probability <= 0.95:
            brightness = random.uniform(-50, 50)
        else:
            brightness = random.uniform(-100, 100)
        brightness_channels = [self.normalizeBrightness(channel, brightness, lower_threshold=5, upper_threshold=250) \
                                for channel in cv2.split(image)]
        return min(brightness_channels)

    def applyContrastAndBrightness(self, image):
        channels = ImageInfo(image).channels
        contrast = self.getContrastParameter(image)
        brightness = self.getBrightnessParameter(image)

        contrasted_channels = [ cv2.multiply(image[:, :, channel_index], contrast) for channel_index in range(channels) ]
        image_brightened_channels = [ cv2.add(contrasted_channels[channel_index], brightness) for channel_index in range(channels) ]

        return cv2.merge(image_brightened_channels)

    def changeContrastAndBrightnessToImage(self, image):
        transformed_image_with_color_noise = self.applyContrastAndBrightness(image[:, :, 0:3])
        return transformed_image_with_color_noise

    def distortImage(self, item_image):
        item_image_info = ImageInfo(item_image)
        (scale_x, scale_y) = self.getScaleParams(item_image_info.width, item_image_info.height)
        scale_matrix = self.getScaleMatrix(scale_x, scale_y)
        rototranslation_matrix = self.getScaledRotoTranslationMatrix(scale_x, scale_y, item_image_info.width, item_image_info.height)
        perspective_matrix = self.getPerspectiveMatrix()
        homography_matrix = np.dot(scale_matrix, np.dot(rototranslation_matrix, perspective_matrix))

        transformed_image = cv2.warpPerspective( item_image, homography_matrix, (constants.input_width, constants.input_height) )
        if item_image_info.channels == 4:
            alpha_channel = transformed_image[:, :, 3]
        if item_image_info.channels == 4:
            return self.image_utils.addAlphaChannelToImage(transformed_image, alpha_channel)
        else:
            return transformed_image
