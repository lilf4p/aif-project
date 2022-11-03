from __future__ import annotations
from salvatore.utils import *
from .base import ContoursLineMetric


class SetDifferenceLinesMetric(ContoursLineMetric):
    """
    Computes a metric for a black-white image where black points represent contours
    from a given target image and the individual is made up by a set of lines, by
    computing simmetric difference between the set of points that represent contours
    and the set of points defined by the lines. The contours are detected using Canny
    edge detector (https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).

    Note that this is equivalent to pixel-based MSE error with appropriate rescaling.
    """
    CHUNKSIZE = 4   # size of chunks to be extracted from individuals

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal,
                 bounds_low: TReal = 0.0, bounds_high: TReal = 1.0):
        """
        :param image_path: Path of the target image.
        :param canny_low: Low threshold for cv2.Canny().
        :param canny_high: High threshold for cv2.Canny().
        :param bounds_low: Lower bounds for representing coordinates. Defaults to 0.0.
        :param bounds_high: Higher bounds for representing coordinates. Defaults to 1.0.
        """
        self.target_contours = set()    # Set of contour points
        self.__target_image_cache = None
        super(SetDifferenceLinesMetric, self).__init__(image_path, canny_low, canny_high, bounds_low, bounds_high)

    def get_target_image(self) -> Image:
        return self.__target_image_cache

    # noinspection PyUnresolvedReferences
    def standardize_target(self):
        pil_image = Image.open(self.image_path)
        cv2_image = pil_to_cv2(pil_image)
        self.image_width, self.image_height = cv2_image.shape[1], cv2_image.shape[0]
        _, contours, _ = find_contours(cv2_image, self.canny_low, self.canny_high)
        # Transform contours from numpy arrays to tuples
        for contour in contours:
            fldim = np.prod(contour.shape)
            contour = np.reshape(contour, (fldim,))
            contour = contour.tolist()
            for i in range(0, len(contour) - 1, 2):
                self.target_contours.add((int(contour[i]), int(contour[i+1])))
        # create and store contour image in memory
        cv2_contour_image = create_monochromatic_image(self.image_width, self.image_height)
        cv2_contour_image = draw_contours(cv2_contour_image, contours, copy=False)
        self.__target_image_cache = Image.fromarray(cv2_contour_image)

    def standardize_individual(self, individual, check_repr=False):
        """
        Expands this individual into a complete sequence of points of lines.
        """
        individual = super(SetDifferenceLinesMetric, self).standardize_individual(individual, check_repr)
        result = []
        for start, end in self.list_to_chunks(individual):
            result += bresenham(start, end, as_list=True)
        return result

    def get_difference(self, individual) -> int:
        # get all individual points as a set
        individual_points = set(self.standardize_individual(individual))
        result = individual_points.symmetric_difference(self.target_contours)
        return len(result)  # number of points that are not contained in both contours and individual points


__all__ = [
    'SetDifferenceLinesMetric',
]
