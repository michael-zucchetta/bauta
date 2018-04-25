import cv2
import numpy as np
import sys
from bauta.utils.ImageUtils import ImageUtils
from bauta.ImageInfo import ImageInfo


class BasicBackgroundRemover():
    def __init__(self):
        self.image_utils = ImageUtils()
        self.object_area_threshold = 0.05

    def findContours(self, filtered_image):
        _, contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) > 0:
            biggest_area_index = np.argmax(areas)
            areas_and_indexes = [(area, index) for index, area in enumerate(areas) if area / areas[biggest_area_index] > self.object_area_threshold]
            sorted_areas_and_indexes = sorted(areas_and_indexes, key=lambda area_index_tuple: area_index_tuple[0], reverse=True)
            area_and_contours = [(area, contours[index]) for (area, index) in sorted_areas_and_indexes]

            return [(area, cv2.approxPolyDP(c, 3, True)) for area, c in area_and_contours]
        else:
            return []

    def applySobelFilter(self, image):
        def sobel(level):
            sobel_horizontal = cv2.Sobel(level, cv2.CV_16S, 1, 0, ksize=3)
            sobel_vertical = cv2.Sobel(level, cv2.CV_16S, 0, 1, ksize=3)
            sobel_response = np.hypot(sobel_horizontal, sobel_vertical)
            sobel_response[sobel_response > 255] = 255
            return sobel_response
        if len(image.shape) == 2:
            sobel_image = sobel(image)
        else:
            sobel_image = np.max( np.array([ sobel(image[:, :, 0]), sobel(image[:, :, 1]), sobel(image[:, :, 2]) ]), axis=0 )
        mean = np.mean(sobel_image)
        sobel_image[sobel_image <= mean] = 0
        sobel_image = sobel_image.astype(np.uint8)
        return sobel_image

    def detectAndRemoveBackgroundColor(self, image):
        image_info = ImageInfo(image)
        color_borders = [image[0][0], image[image_info.height - 1][0], image[0][image_info.width - 1], image[image_info.height - 1][image_info.width - 1]]
        result = 255
        for color_border in color_borders:
            result = cv2.subtract( result, cv2.inRange(image, cv2.subtract(color_border, 2), cv2.add(color_border, 1) ))
        result = cv2.GaussianBlur(result, (5, 5), 3)
        result = cv2.erode(result, None, iterations=2)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, None, iterations=3)
        areas_and_contours = self.findContours(result)
        cv2.drawContours(result, [areas_and_contours[0][1]], -1, (255), -1)
        for i in range(1, len(areas_and_contours)):
            (area, contour) = areas_and_contours[i]
            if area / areas_and_contours[0][0] > self.object_area_threshold:
                color = 0
            else:
                color = 255
            cv2.drawContours(result, [areas_and_contours[i][1]], -1, color, -1)
        return result 

    def removeBackgroundInsideMainObject(self, original_image, contours, mask):
        original_image_info = ImageInfo(original_image)
        biggest_area, _ = contours[0]
        background_subtracted_mask  = self.detectAndRemoveBackgroundColor(original_image)
        for contour_index in range(1, len(contours)):
            (area, contour) = contours[contour_index]
            first_pixel_in_contour = contour[0][0]
            first_pixel_in_contour = (first_pixel_in_contour[1], first_pixel_in_contour[0])
            # if the contour is at least 50% the size of the biggest element => it iss probably another object if it is not overlapping
            if area / biggest_area > 0.5 and not background_subtracted_mask[first_pixel_in_contour] == 0:
                cv2.fillPoly(mask, [contour], 255)
            else:
                '''
                checking if the contour is a background or not. If the background_subtracted_mask has values as zero that are inside the contour, then the contour
                is part of the background
                '''
                if contour_index == 1:
                    background_subtracted_mask[background_subtracted_mask == 1] = 3
                    background_subtracted_mask[background_subtracted_mask == 2] = 3
                contour_canvas = self.image_utils.blankImage(original_image_info.width, original_image_info.height)
                cv2.fillPoly(contour_canvas, [contour], 1)
                overlapping_areas = cv2.add(background_subtracted_mask, contour_canvas)
                mask[overlapping_areas == 1] = 0

    def removeFlatBackgroundFromRGB(self, image, full_computation=True):
        if len(image.shape) > 2 and image.shape[2] == 4:
            '''
            if the alpha channel contains at least one pixel that is not fully white,
            then the alpha channel is truly an alpha channel
            '''
            if np.any(image[:, :, 3] != 255):
                return image
            else:
                (b, g, r, _) = cv2.split(image)
                image = cv2.merge([b, g, r])
        image_info = ImageInfo(image)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        sobel_image = self.applySobelFilter(blurred_image)
        contours = self.findContours(sobel_image)
        mask = self.image_utils.blankImage(image_info.width, image_info.height)
        if len(contours) == 0:
            contours = [(0, np.array([[0, 0], [0, image_info.height- 1], [image_info.width- 1, 0], [image_info.width - 1, image_info.height - 1]]))]
        _, biggest_contour = contours[0]
        cv2.fillPoly(mask, [biggest_contour], 255)
        if full_computation and len(contours) > 1:
            self.removeBackgroundInsideMainObject(blurred_image, contours, mask)
        mask = cv2.erode(mask, None, iterations=4)
        b, g, r = cv2.split(image)
        rgba = [b, g, r, mask]
        rgba = cv2.merge(rgba, 4)
        return rgba
