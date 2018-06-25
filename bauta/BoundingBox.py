import cv2
import math

class BoundingBox():

    def __init__(self, top, left, bottom, right):
        if(top > bottom ):
            error_message = f"Bounding Box (top: {top}, left: {left}, bottom: {bottom}, right: {right}) cannot be created. Bottom({bottom}) should be smaller or equal than top({top})"
            raise Exception(error_message)
        if(left > right ):
            error_message = f"Bounding Box (top: {top}, left: {left}, bottom: {bottom}, right: {right}) cannot be created. Left({left}) should be smaller or equal than right({right})"
            raise Exception(error_message)
        self._top    = int(top)
        self._left   = int(left)
        self._bottom = int(bottom)
        self._right  = int(right)
        self._width  = self._right  - self._left + 1
        self._height = self._bottom - self._top + 1
        self._area   = self._width  * self._height

    def cropImage(self, image, x_delta=0, y_delta=0):
        return image[self.top - y_delta:self.top+self.height + y_delta,self.left-x_delta:self.left+self.width+x_delta,:]

    def fromOpenCVConnectedComponentsImage(image, min_threshold=0.25, max_threshold=1):
        ret, image = cv2.threshold(image, min_threshold, max_threshold, cv2.THRESH_BINARY)
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)
        return BoundingBox.fromOpenCVConnectedComponents(num_labels, labels, stats, centroids)

    def fromOpenCVConnectedComponents(num_labels, labels, stats, centroids):
        bounding_boxes = []
        countour_areas = []
        for label_index in range(0, min(labels.shape[0], stats.shape[0])):
            countour_area = stats[label_index, cv2.CC_STAT_AREA]
            countour_areas.append(countour_area)
            label  = labels[label_index]
            top    = stats[label_index, cv2.CC_STAT_TOP]
            left   = stats[label_index, cv2.CC_STAT_LEFT]
            bottom = top + stats[label_index, cv2.CC_STAT_HEIGHT] - 1
            right  = left + stats[label_index, cv2.CC_STAT_WIDTH] - 1
            if top < bottom and left < right:
                bounding_box = BoundingBox(top, left, bottom, right)
                bounding_boxes.append(bounding_box)
        return bounding_boxes, countour_areas

    def intersectionOverUnion(self, bounding_box):
        intersection_area = self.intersectingArea(bounding_box)
        return intersection_area / (self.area + bounding_box.area - intersection_area)

    def intersect(self, bounding_box):
        intersection_left   = max(self.left, bounding_box.left)
        intersection_right  = min(self.right, bounding_box.right)
        intersection_bottom = min(self.bottom, bounding_box.bottom)
        intersection_top    = max(self.top, bounding_box.top)
        if intersection_left <= intersection_right and intersection_top <= intersection_bottom:
            return BoundingBox(intersection_top, intersection_left, intersection_bottom, intersection_right)
        else:
            return None

    def intersectingArea(self, bounding_box):
        intersection = self.intersect(bounding_box)
        if(intersection is None):
            return 0
        else:
            return intersection.area

    def resize(self, from_width, from_height, to_width, to_height):
        width_aspect_ratio = to_width / from_width
        height_aspect_ratio = to_height / from_height
        return BoundingBox( int(self._top    * height_aspect_ratio),
                            int(self._left   * width_aspect_ratio),
                            min(to_height - 1, int(self._bottom * height_aspect_ratio) + math.ceil(height_aspect_ratio) - 1),
                            min(to_width - 1, int(self._right  * width_aspect_ratio) + math.ceil(width_aspect_ratio) - 1)
                            )
    def cropTensor(self, tensor_to_crop, batch):
        return tensor_to_crop[batch:batch+1, :,self.top:self.bottom+1,self.left:self.right+1]   

    def areBoundsAproximatelySimilar(self, bounding_box, pixels_interval=10):      
        return ( self._top <= bounding_box._top + pixels_interval or self._top >= bounding_box._top - pixels_interval ) and \
            ( self._left <= bounding_box._left + pixels_interval or self._left >= bounding_box._left - pixels_interval ) and \
            ( self._right <= bounding_box._right + pixels_interval or self._right >= bounding_box._right - pixels_interval ) and \
            ( self._bottom <= bounding_box._bottom + pixels_interval or self._bottom >= bounding_box._bottom - pixels_interval )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._top == other._top and self._left == other._left and self._bottom == other._bottom and self._right == other._right
        return False

    def __hash__(self):
        return hash((self._top, self._left, self._right, self._bottom))

    def __str__(self):
        return f"(top: {self._top}, left: {self._left}, bottom: {self._bottom}, right: {self._right})"

    def __repr__(self):
        return str(self)

    @property
    def top(self):
        return self._top

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def bottom(self):
        return self._bottom

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def area(self):
        return self._area
