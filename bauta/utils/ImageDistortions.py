import cv2
import math
import numpy as np
import random

from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo

class ImageDistortions():
    def __init__(self):
        self.image_utils = ImageUtils()

    def _generateRotationDistortionParameters():
        return (-random.random() * 20) + 10

    def applyRotationDistortion(self, item_image, angle_distortion=_generateRotationDistortionParameters()):
        item_image_info = ImageInfo(item_image)
        extra_rotation_border = math.ceil( math.sqrt(item_image_info.width ** 2 + item_image_info.height ** 2 / 4) )
        transparent_background_width = int(item_image_info.width + extra_rotation_border)
        transparent_background_height = int(item_image_info.height + extra_rotation_border)
        transparent_background_for_item_image_rgb = self.image_utils.blankImage(transparent_background_width, transparent_background_height, 4)
        # item is pasted in the center of a transparent background which is 50% bigger than itself
        (item_image_on_bigger_tranparent_image, mask) = self.image_utils.pasteRGBAimageIntoRGBimage(item_image, transparent_background_for_item_image_rgb, int((transparent_background_width - item_image_info.width) / 2),  int((transparent_background_height - item_image_info.height) / 2))
        item_image_on_bigger_tranparent_image[:, :, 3] = mask[:, :, 0]
        rotation_matrix = cv2.getRotationMatrix2D( center=(int(transparent_background_width / 2), int(transparent_background_height / 2) ), angle=angle_distortion, scale=1)
        rotated_item_image = cv2.warpAffine(item_image_on_bigger_tranparent_image, rotation_matrix, (transparent_background_width, transparent_background_height))
        rotated_item_image = cv2.resize(rotated_item_image, (item_image_info.width,  item_image_info.height))
        return rotated_item_image

    def applyTransformationsAndDistortions(self, item_image):
        item_image = self.applyRotationDistortion(item_image)

        return item_image
