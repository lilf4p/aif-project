from __future__ import annotations
from salvatore.utils import *
from .base import ContoursMetric, ABContoursMetric, AContoursMetric


class NearestNeighbourPointMetric(ContoursMetric):
    """
    A metric that, given a population of points with coordinates (x, y), computes
    the fitness as the sum of the minimum Manhattan distances from each point to
    the target set of points.
    """
    CHUNKSIZE = 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal,
                 bounds_low: TReal = 0.0, bounds_high: TReal = 1.0, num_points: int = 20000,
                 device='cpu'):
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
        super(NearestNeighbourPointMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device)

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
        aux = self.vp.zeros((tot_contours, 2))
        i = 0
        for contour in contours_list:
            contour = contour if self.device == 'cpu' else self.vp.array(contour)
            k = len(contour)
            aux[i:i+k, :] = contour
            i += k
        # now aux contains a column with all the widths of the target points and another one with the heights
        self.target = self.vp.zeros((2, tot_contours, self.num_points))  # on GPU
        # self.target[0, :, :] = aux[:, 0]
        # self.target[1, :, :] = aux[:, 1]
        for j in range(self.num_points):
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

    def check_individual_repr(self, individual) -> TBoolStr:
        """
        Individual should be represented as a list with length a multiple of 4, where each
        disjoint subsequence of 4 elements from the start represent coordinates of the two
        extremes of the line.
        """
        if not isinstance(individual, Sequence):
            return False, f"Invalid type for individual: Sequence, got {type(individual)}"
        ilen = len(individual)
        if ilen != 2 * self.num_points:
            return False, f"Invalid length for individual sequence: expected {2 * self.num_points}, got {ilen}"
        for i in range(ilen):
            ith = individual[i]
            if not isinstance(ith, int) or not isinstance(ith, float):
                return False, f"Invalid object at index {i}: expected int or float, got {type(ith)}"
            if not (self.bounds_low <= ith <= self.bounds_high):
                return False, f"Invalid value {ith} at index {i}"
        return True, None

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

    def standardize_individual(self, individual, check_repr=False):
        scaled = [[], []]   # 2 rows x num_points columns
        for w, h in self.__list_to_chunks(individual):
            scaled[0] += [w]
            scaled[1] += [h]
        aux = self.vp.array(scaled)
        target_individual = self.vp.zeros((2, self.num_targets, self.num_points))
        for i in range(self.num_targets):
            target_individual[:, i] = self.vp.abs(aux[:] - self.target[:, i])
        return target_individual

    def _core_get_difference(self, individual):
        standardized = self.standardize_individual(individual)
        summed = self.vp.sum(standardized, axis=0)
        del standardized
        # target_individual = cp.abs(target_individual - self.target)
        # target_individual[0] += target_individual[1]
        target_individual = self.vp.min(summed, axis=0)  # todo reset to 1 afterwards!
        del summed
        result = self.vp.sum(target_individual)
        del target_individual
        return result

    # @timeit('fitnesses')
    def get_difference(self, individual):
        if self.device == 'cpu':
            return self._core_get_difference(individual)
        else:
            with cp.cuda.Stream() as stream:
                result = self._core_get_difference(individual)
            stream.synchronize()
            return result.asnumpy()
        # return result if self.device == 'cpu' else self.vp.asnumpy(result)


class ABNearestNeighbourPointMetric(AContoursMetric):
    """
    A metric that, given a population of points with coordinates (x, y), computes
    the fitness as the sum of the minimum Manhattan distances from each point to
    the target set of points.
    """
    CHUNKSIZE = 2

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
                 bounds_high: TReal = 1.0, num_points: int = 20000, device='cpu', distance_type='min'):
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
        self.distance_type = distance_type
        super(ABNearestNeighbourPointMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device
        )
        self.fitness_calc = self._get_fitness_calc()
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

    @staticmethod
    def _gpu_fitness_calc_max(x: cp.ndarray, y: cp.ndarray):
        return cp.sum(cp.max(cp.sum(cp.abs(x-y), axis=0), axis=1))

    def cpu_fitness_calc_max(self, x: np.ndarray, y: np.ndarray, index):
        result = np.sum(np.max(np.sum(np.abs(x-y), axis=0), axis=1))
        self.results[index] = result

    def gpu_fitness_calc_max(self, x: cp.ndarray, y: cp.ndarray, index):
        self.results[index] = self._gpu_fitness_calc_max(x, y)

    @staticmethod
    def _gpu_fitness_calc_avg(x: cp.ndarray, y: cp.ndarray):
        return cp.sum(cp.mean(cp.sum(cp.abs(x-y), axis=0), axis=1))

    def cpu_fitness_calc_avg(self, x: np.ndarray, y: np.ndarray, index):
        result = np.mean(np.mean(np.sum(np.abs(x-y), axis=0), axis=1))
        self.results[index] = result

    def gpu_fitness_calc_avg(self, x: cp.ndarray, y: cp.ndarray, index):
        self.results[index] = self._gpu_fitness_calc_avg(x, y)

    def _get_fitness_calc(self):
        if self.distance_type == 'min':
            return self.cpu_fitness_calc_min if self.device == 'cpu' else self.gpu_fitness_calc_min
        elif self.distance_type == 'max':
            return self.cpu_fitness_calc_max if self.device == 'cpu' else self.gpu_fitness_calc_max
        elif self.distance_type in ['avg', 'mean']:
            return self.cpu_fitness_calc_avg if self.device == 'cpu' else self.gpu_fitness_calc_avg
        else:
            raise ValueError(f"Unknown distance type {self.distance_type}")

    def _core_get_difference(self, individual: TArray, index: int = 0):
        standardized = self.standardize_individual(individual)
        self.fitness_calc(standardized, self.target, index)

    # @timeit('get_difference')
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
                stream.synchronize()    # fixme check if sync at the end of each iteration fixes!
        results = self.results[:n_ind] if self.device == 'cpu' else cp.asnumpy(self.results[:n_ind])  # fixme check if this alters someway
        return results.copy()   # fixme check if this is necessary


__all__ = [
    'NearestNeighbourPointMetric',
    'ABNearestNeighbourPointMetric',
]
