import cv2
class ImageInfo():

    def __init__(self, image):
        self.height = image.shape[0]
        self.width  = image.shape[1]
        self.aspect_ratio = self.width / self.height
        self.channels = image.shape[2]

    def __hash__(self):
        return hash((self.height, self.width, self.channels))

    def __str__(self):
        return f"(width: {self.width}, height: {self.height}, channels: {self.channels}, aspect ratio: {self.aspect_ratio:{1}.{4}})"

    def __repr__(self):
        return str(self)
