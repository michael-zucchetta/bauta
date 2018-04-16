class InferenceResult():

    def __init__(self, class_label, bounding_box, mask, contour_area=None, image=None):
        self.class_label = class_label
        self.bounding_box  = bounding_box
        self.mask = mask
        self.contour_area = contour_area
        self.image = image

    def __str__(self):
        return f"(class_label: {self.class_label}, bounding_box: {self.bounding_box})"

    def __repr__(self):
        return str(self)
