from __future__ import annotations
from salvatore.utils import *
from .base import AContoursMetric


class TargetPointsArrayNearestNeighbourPointMetric(AContoursMetric):
    """
    A metric that, given a population of points with coordinates (x, y), computes
    the fitness as the sum of the minimum Manhattan distances from each target point
    to the set of candidate points.
    """
    CHUNKSIZE = 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
                 bounds_high: TReal = 1.0, num_points: int = 20000, device='cpu'):
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
        self.vp = np if device == 'cpu' else cp
        super(TargetPointsArrayNearestNeighbourPointMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device
        )
        self.fitness_calc = self.cpu_fitness_calc_min if device == 'cpu' else self.gpu_fitness_calc_min
        self.results = self.vp.zeros(self.num_targets)
        self.target_individuals = self.vp.zeros((2, self.num_targets, self.num_points))

    def get_target_image(self) -> Image:
        return self.target_pil

    # noinspection PyUnresolvedReferences
    def standardize_target(self):
        pil_image = Image.open(self.image_path)
        cv2_image = pil_to_cv2(pil_image)
        self.image_width, self.image_height = cv2_image.shape[1], cv2_image.shape[0]

        # find contours
        _, contours, _ = find_contours(cv2_image, self.canny_low, self.canny_high)

        # Transform contours from numpy arrays to tuples
        elapsed_time = time()
        tot_contours = 0
        contours_list = []
        for contour in contours:
            fldim = np.prod(contour.shape)
            contour = np.reshape(contour, (fldim // 2, 2))
            tot_contours += len(contour)
            contours_list.append(contour)
        # objective tensor: (2, num_targets, num_points)
        self.num_targets = tot_contours
        aux = np.zeros((tot_contours, 2))
        i = 0
        for contour in contours_list:
            # contour = contour if self.device == 'cpu' else self.vp.array(contour)
            k = len(contour)
            aux[i:i+k, :] = contour
            i += k
        # now aux contains a column with all the widths of the target points and another one with the heights
        # and it will be transferred to gpu if specified in constructor
        # aux = aux.astype(np.int32)  # todo pass integer type as parameter!
        aux = aux if self.device == 'cpu' else self.vp.asarray(aux)
        self.target = self.vp.zeros((2, tot_contours, self.num_points))
        for j in range(self.num_points):    # todo check if this double slice works!
            self.target[0, :, j] = aux[:, 0]
            self.target[1, :, j] = aux[:, 1]
        elapsed_time = time() - elapsed_time
        print(elapsed_time)
        # create and store contour image in memory
        target_cv2 = create_monochromatic_image(self.image_width, self.image_height)
        target_cv2 = draw_contours(target_cv2, contours, copy=False)
        self.target_pil = Image.fromarray(target_cv2)

        # cleanup
        del contours_list, aux, target_cv2

    def __list_to_chunks(self, individual):
        """
        Breaks individual into a sequence of couples of tuples
        ((start_x, start_y), (end_x, end_y)) already rescaled.
        """
        for chunk in range(0, len(individual), self.CHUNKSIZE):
            yield int(individual[chunk] * self.image_width), int(individual[chunk+1] * self.image_height)

    def get_individual_image(self, individual) -> Image:
        image = Image.new('F', (self.image_width, self.image_height), color=255)
        draw = ImageDraw.Draw(image, 'F')
        for x, y in self.__list_to_chunks(individual):
            draw.point((x, y), fill=0)
        # cleanup
        del draw
        return image

    def standardize_individual(self, individual: TArray):
        # reshape and rescale
        reshaped = np.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height

        # reshaped = reshaped.astype(np.int32)
        reshaped = reshaped if self.device == 'cpu' else cp.asarray(reshaped)
        self.target_individuals[0, :] = reshaped[:, 0]
        self.target_individuals[1, :] = reshaped[:, 1]
        return self.target_individuals

    @staticmethod
    def _gpu_fitness_calc_min(x: cp.ndarray, y: cp.ndarray):
        return cp.sum(cp.min(cp.sum(cp.abs(x-y), axis=0), axis=1))

    def cpu_fitness_calc_min(self, x: np.ndarray, y: np.ndarray, index):
        result = np.sum(np.min(np.sum(np.abs(x-y), axis=0), axis=1))
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
        return results.copy()   # fixme check if this is necessary


class TableTargetPointsNNContoursMetric(AContoursMetric):
    CHUNKSIZE = 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
                 bounds_high: TReal = 1.0, num_points: int = 20000, device='cpu'):
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
        self.vp = np if device == 'cpu' else cp
        super(TableTargetPointsNNContoursMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device
        )
        self.results = self.vp.zeros(self.num_targets, dtype=self.vp.int32)
        self.target_individuals = self.vp.zeros((2, self.num_targets, self.num_points))

    def get_target_image(self) -> Image:
        return self.target_pil

    # noinspection PyUnresolvedReferences
    def standardize_target(self):
        pil_image = Image.open(self.image_path)
        cv2_image = pil_to_cv2(pil_image)
        self.image_width, self.image_height = cv2_image.shape[1], cv2_image.shape[0]

        # find contours
        _, contours, _ = find_contours(cv2_image, self.canny_low, self.canny_high)

        # Transform contours from numpy arrays to tuples
        tot_contours = 0
        contours_list = []
        for contour in contours:
            fldim = np.prod(contour.shape)
            contour = np.reshape(contour, (fldim // 2, 2))
            tot_contours += len(contour)
            contours_list.append(contour)
        # objective tensor: (2, num_targets, num_points)
        self.num_targets = tot_contours
        aux = np.zeros((tot_contours, 2))
        i = 0
        for contour in contours_list:
            # contour = contour if self.device == 'cpu' else self.vp.array(contour)
            k = len(contour)
            aux[i:i+k, :] = contour
            i += k
        # now aux contains a column with all the widths of the target points and another one with the heights
        # and it will be transferred to gpu if specified in constructor
        # aux = aux.astype(np.int32)  # todo pass integer type as parameter!
        aux = aux if self.device == 'cpu' else self.vp.asarray(aux)
        target_aux_0 = self.vp.zeros((2, tot_contours, self.image_width))
        target_aux_1 = self.vp.zeros((2, tot_contours, self.image_width))
        for j in range(self.image_width):    # todo check if this double slice works!
            target_aux_0[0, :, j] = aux[:, 0]
            target_aux_0[1, :, j] = aux[:, 1]
            target_aux_1[0, :, j] = aux[:, 0]
            target_aux_1[1, :, j] = aux[:, 1]

        # Build table of distances
        self.target = self.vp.full((self.image_height, self.image_width), -1, dtype=np.int32)
        for i in range(self.image_height):
            target_aux_1[1] -= i
            for j in range(self.image_width):  # todo questo si può modificare con un array (self.width,) creato appositamente!
                target_aux_1[0, :, j] -= j
            target_aux_1 = np.abs(target_aux_1)
            target_aux_2 = np.sum(target_aux_1, axis=0)
            result = np.min(target_aux_2, axis=0)
            # result = np.min(np.sum(np.abs(target_aux_1), axis=0), axis=0)
            self.target[i, :] = result
            self.vp.copyto(dst=target_aux_1, src=target_aux_0)

        # create and store contour image in memory
        target_cv2 = create_monochromatic_image(self.image_width, self.image_height)
        target_cv2 = draw_contours(target_cv2, contours, copy=False)
        self.target_pil = Image.fromarray(target_cv2)

        # cleanup
        del contours_list, aux, target_cv2

    def __list_to_chunks(self, individual):
        """
        Breaks individual into a sequence of couples of tuples
        ((start_x, start_y), (end_x, end_y)) already rescaled.
        """
        for chunk in range(0, len(individual), self.CHUNKSIZE):
            yield int(individual[chunk] * self.image_width), int(individual[chunk+1] * self.image_height)

    def get_individual_image(self, individual) -> Image:
        image = Image.new('F', (self.image_width, self.image_height), color=255)
        draw = ImageDraw.Draw(image, 'F')
        for x, y in self.__list_to_chunks(individual):
            draw.point((x, y), fill=0)
        # cleanup
        del draw
        return image

    def standardize_individual(self, individual: TArray):
        # reshape and rescale
        individual = individual if self.device == 'cpu' else cp.asarray(individual)
        reshaped = self.vp.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        return reshaped.astype(dtype=self.vp.int32).T

    def _core_get_difference(self, individual: TArray, index: int = 0):
        standardized = self.standardize_individual(individual)
        aux = self.target[standardized[1], standardized[0]]
        self.results[index] = self.vp.sum(aux)

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
        return results.copy()   # fixme check if this is necessary


class DoubleArrayNearestNeighbourPointMetric(TargetPointsArrayNearestNeighbourPointMetric):
    """
    A metric that, given a population of points with coordinates (x, y), computes
    the fitness as the sum of the minimum Manhattan distances from both each target
    point to the set of candidate points and vice versa.
    """
    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
                 bounds_high: TReal = 1.0, num_points: int = 20000, device='cpu',
                 target_candidate_weight: TReal = 2.0, candidate_target_weight: TReal = 1.0):
        """
        :param target_candidate_weight: Weight to assign to the part of the fitness function
        that calculates the distance between each target point and its nearest candidate one.
        Defaults to 2.0, and it should have a higher value than candidate_target_weight, since
        in general this distance sum is much smaller than the one calc'd by candidate->target.
        """
        super(DoubleArrayNearestNeighbourPointMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, num_points, device
        )
        self.tc_weight = target_candidate_weight
        self.ct_weight = candidate_target_weight

    # Min distance
    def _gpu_fitness_calc_min(self, x: cp.ndarray, y: cp.ndarray):
        z = cp.sum(cp.abs(x-y), axis=0)
        return self.tc_weight * cp.sum(cp.min(z, axis=1)) + self.ct_weight * cp.sum(cp.min(z, axis=0))

    def cpu_fitness_calc_min(self, x: np.ndarray, y: np.ndarray, index):
        z = np.sum(np.abs(x-y), axis=0)
        result = self.tc_weight * np.sum(np.min(z, axis=1)) + self.ct_weight * np.sum(np.min(z, axis=0))
        self.results[index] = result

    def gpu_fitness_calc_min(self, x: cp.ndarray, y: cp.ndarray, index):
        self.results[index] = self._gpu_fitness_calc_min(x, y)


__all__ = [
    'TargetPointsArrayNearestNeighbourPointMetric',
    'TableTargetPointsNNContoursMetric',
    'DoubleArrayNearestNeighbourPointMetric',
]
