from __future__ import annotations
from salvatore.utils import *
from .base import ArrayPointContoursMetric
from .utils import *


class TableTargetPointsNNContoursMetric(ArrayPointContoursMetric):

    @property
    def CHUNKSIZE(self):
        return 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
                 bounds_high=1.0, results=None, num_points: int = 20000):
        """
        :param image_path: Path of the target image.
        :param canny_low: Low threshold for cv2.Canny().
        :param canny_high: High threshold for cv2.Canny().
        :param bounds_low: Lower bounds for representing coordinates. Defaults to 0.0.
        :param bounds_high: Higher bounds for representing coordinates. Defaults to 1.0.
        """
        self.target = None  # target image as a (2, num_contours, num_points) tensor of integers
        self.target_pil = None  # target image as PIL.Image object
        self.num_targets = 0
        self.num_points = num_points
        super(TableTargetPointsNNContoursMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, results=results,
        )

    def check_individual_repr(self, individual) -> TBoolStr:
        return True, None   # no interest in actual checking (by now)

    def standardize_target(self):
        aux, pil_image, cv2_image, contours = extract_contours(self.image_path, self.canny_low, self.canny_high)
        self.image_width, self.image_height = cv2_image.shape[1], cv2_image.shape[0]
        self.num_targets = len(aux)
        self.target = build_distance_table(self.image_height, self.image_width, self.num_targets, aux)
        # create and store contour image in memory
        target_cv2 = create_monochromatic_image(self.image_width, self.image_height)
        target_cv2 = draw_contours(target_cv2, contours, copy=False)
        self.target_pil = Image.fromarray(target_cv2)

    def standardize_individual(self, individual, check_repr=False):
        reshaped = individual.copy()
        reshaped = np.reshape(reshaped, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        return reshaped.astype(dtype=np.intp).T

    def _core_get_difference(self, individual: TArray, index: int = 0):
        standardized = self.standardize_individual(individual)
        aux = self.target[standardized[1], standardized[0]]
        self.results[index] = np.sum(aux)


__all__ = [
    'TableTargetPointsNNContoursMetric',
]
