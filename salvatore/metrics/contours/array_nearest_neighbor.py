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

    def standardize_individual(self, individual: TArray, check_repr=False):
        # reshape and rescale
        reshaped = np.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        return reshaped.astype(dtype=np.int32).T

    def _core_get_difference(self, individual: TArray, index: int = 0):
        standardized = self.standardize_individual(individual)
        aux = self.target[standardized[1], standardized[0]]
        self.results[index] = np.sum(aux)


class DoubleArrayNearestNeighbourPointMetric(ArrayPointContoursMetric):
    """
    A metric that, given a population of points with coordinates (x, y), computes
    the fitness as the sum of the minimum Manhattan distances from both each target
    point to the set of candidate points and vice versa.
    """

    @property
    def CHUNKSIZE(self):
        return 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low=0.0,
                 bounds_high=1.0, num_points: int = 20000, results=None, device='cpu',
                 target_candidate_weight=2.0, candidate_target_weight=1.0):
        """
        :param target_candidate_weight: Weight to assign to the part of the fitness function
        that calculates the distance between each target point and its nearest candidate one.
        Defaults to 2.0, and it should have a higher value than candidate_target_weight, since
        in general this distance sum is much smaller than the one calc'd by candidate->target.
        """
        self.target = None  # target image as a (2, num_contours, num_points) tensor of integers
        self.target_pil = None  # target image as PIL.Image object
        self.num_targets = 0
        self.num_points = num_points
        self.vp = np if device == 'cpu' else cp
        self.device = device
        # results = self.vp.asarray(results) if results is not None else results
        super(DoubleArrayNearestNeighbourPointMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, results=results,
        )
        self.fitness_calc = self.cpu_fitness_calc_min if device == 'cpu' else self.gpu_fitness_calc_min
        self.target_individuals = self.vp.zeros((2, self.num_targets, self.num_points))
        self.tc_weight = target_candidate_weight
        self.ct_weight = candidate_target_weight

    # Min distance
    def standardize_target(self):
        aux, pil_image, cv2_image, contours = extract_contours(
            self.image_path, self.canny_low, self.canny_high, self.device
        )
        self.image_width, self.image_height = cv2_image.shape[1], cv2_image.shape[0]
        self.target = self.vp.zeros((2, len(aux), self.num_points))
        for j in range(self.num_points):    # todo check if this double slice works!
            self.target[0, :, j] = aux[:, 0]
            self.target[1, :, j] = aux[:, 1]
        # create and store contour image in memory
        target_cv2 = create_monochromatic_image(self.image_width, self.image_height, device='cpu')
        target_cv2 = draw_contours(target_cv2, contours, copy=False)
        self.target_pil = Image.fromarray(target_cv2)
        self.num_targets = len(aux)

    def check_individual_repr(self, individual) -> TBoolStr:
        return True, None   # no interest in actual checking (by now)

    def standardize_individual(self, individual: TArray, check_repr=False):
        # reshape and rescale
        reshaped = np.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        reshaped = self.vp.asarray(reshaped)
        self.target_individuals[0, :] = reshaped[:, 0]
        self.target_individuals[1, :] = reshaped[:, 1]
        return self.target_individuals

    def _gpu_fitness_calc_min(self, x: cp.ndarray, y: cp.ndarray):
        z = cp.sum(cp.abs(x-y), axis=0)
        return self.tc_weight * cp.sum(cp.min(z, axis=1)) + self.ct_weight * cp.sum(cp.min(z, axis=0))

    def cpu_fitness_calc_min(self, x: np.ndarray, y: np.ndarray, index):
        z = np.sum(np.abs(x-y), axis=0)
        result = self.tc_weight * np.sum(np.min(z, axis=1)) + self.ct_weight * np.sum(np.min(z, axis=0))
        self.results[index] = result

    def gpu_fitness_calc_min(self, x: cp.ndarray, y: cp.ndarray, index):
        self.results[index] = self._gpu_fitness_calc_min(x, y)

    def _core_get_difference(self, individual: TArray, index: int = 0):
        standardized = self.standardize_individual(individual)
        # noinspection PyArgumentList
        self.fitness_calc(standardized, self.target, index)

    def get_difference(self, individuals: TArray):
        n_ind = len(individuals)
        self.results[:] = 0
        if self.device == 'cpu':
            for index in range(n_ind):
                self._core_get_difference(individuals[index], index)
        else:
            with cp.cuda.Stream() as stream:
                for index in range(n_ind):
                    self._core_get_difference(individuals[index], index)
                stream.synchronize()
        results = self.results[:n_ind] if self.device == 'cpu' else cp.asnumpy(self.results[:n_ind])
        return results,   # fixme check if this is necessary


class TableTargetPointsOverlapPenaltyContoursMetric(ArrayPointContoursMetric):

    @property
    def CHUNKSIZE(self):
        return 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal,
                 bounds_low=0.0, bounds_high=1.0, results=None, num_points: int = 20000):
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
        super(TableTargetPointsOverlapPenaltyContoursMetric, self).__init__(
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

    def standardize_individual(self, individual: TArray, check_repr=False):
        # reshape and rescale
        reshaped = np.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        return reshaped.astype(dtype=np.int32).T

    def _core_get_difference(self, individual: TArray,index: int = 0):
        standardized = self.standardize_individual(individual)
        aux = self.target[standardized[1], standardized[0]]
        self.results[index] = np.sum(aux)
        img = np.zeros((self.image_height, self.image_width), dtype=np.int32)
        for i in range(standardized.shape[1]):
            w, h = standardized[0, i], standardized[1, i]
            img[h, w] += 1
        img[img > 0] -= 1
        penalty = img.sum()
        self.results[index] += penalty


__all__ = [
    'TableTargetPointsNNContoursMetric',
    'DoubleArrayNearestNeighbourPointMetric',
    'TableTargetPointsOverlapPenaltyContoursMetric',
]
