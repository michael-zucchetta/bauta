import unittest

from bauta.BoundingBox import BoundingBox

class TestBoundingBox(unittest.TestCase):

    def test_dimensions(self):
        bounding_box = BoundingBox(-10, 5, 4, 6)
        self.assertEqual(bounding_box.area, 30)
        self.assertEqual(bounding_box.width, 2)
        self.assertEqual(bounding_box.height, 15)

    def test_intersect_no_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(-10, 0, 4, 4)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), None)
        bounding_box_to_intersect = BoundingBox(16, 16, 17, 17)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), None)

    def test_intersect_top_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(-10, 0, 5, 20)
        bounding_box_expected_intersection = BoundingBox(5, 5, 5, 15)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), bounding_box_expected_intersection)

    def test_intersect_bottom_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(15, -10, 16, 20)
        bounding_box_expected_intersection = BoundingBox(15, 5, 15, 15)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), bounding_box_expected_intersection)

    def test_intersect_left_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(0, -10, 20, 5)
        bounding_box_expected_intersection = BoundingBox(5, 5, 15, 5)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), bounding_box_expected_intersection)

    def test_intersect_right_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(-10, 15, 20, 16)
        bounding_box_expected_intersection = BoundingBox(5, 15, 15, 15)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), bounding_box_expected_intersection)

    def test_intersect_diagonal_interesction(self):
        bounding_box = BoundingBox(5, 5, 15, 15)
        bounding_box_to_intersect = BoundingBox(-10, -15, 6, 6)
        bounding_box_expected_intersection = BoundingBox(5, 5, 6, 6)
        self.assertEqual(bounding_box.intersect(bounding_box_to_intersect), bounding_box_expected_intersection)

if __name__ == '__main__':
    unittest.main()
