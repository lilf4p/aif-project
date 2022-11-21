# A combination of the "nearest-neighbour" approach with points and MSE for lines
from salvatore.utils import *
from .base import ContoursLineMetric


class LinesNNPointContoursMetric(ContoursLineMetric):

    @property
    def chunk_size(self):
        return 4

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal, bounds_low: TReal = 0.0,
                 bounds_high: TReal = 1.0, lineno: int = 500, device='cpu',
                 point_adherence_coeff: float = 10.0, line_adherence_coeff: float = 1.0,
                 line_l1_lambda: float = 5.0):
        """
        :param image_path: Path of the target image.
        :param canny_low: Low threshold for cv2.Canny().
        :param canny_high: High threshold for cv2.Canny().
        :param bounds_low: Lower bounds for representing coordinates. Defaults to 0.0.
        :param bounds_high: Higher bounds for representing coordinates. Defaults to 1.0.
        :param device: Device to use for calculating fitness, either 'cpu' or 'gpu'. Defaults to 'cpu'.
        """
        self.target_cv2 = None  # target image in cv2 (array) format
        self.target_table = None  # target image as table of distances
        self.target_pil = None  # target image as PIL.Image object
        self.device = device
        self.vp = np if self.device == 'cpu' else cp
        self.num_targets = 0
        self.num_points = lineno * (self.chunk_size // 2)
        self.point_adherence_coeff = point_adherence_coeff
        self.line_adherence_coeff = line_adherence_coeff
        self.line_l1_lambda = line_l1_lambda
        super(LinesNNPointContoursMetric, self).__init__(
            image_path, canny_low, canny_high, bounds_low, bounds_high, device=device,
        )
        self.results = self.vp.zeros(self.num_targets, dtype=self.vp.int32)

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
        self.target_table = self.vp.full((self.image_height, self.image_width), -1, dtype=np.int32)
        for i in range(self.image_height):
            target_aux_1[1] -= i
            for j in range(self.image_width):  # todo questo si può modificare con un array (self.width,) creato appositamente!
                target_aux_1[0, :, j] -= j
            target_aux_1 = np.abs(target_aux_1)
            target_aux_2 = np.sum(target_aux_1, axis=0)
            result = np.min(target_aux_2, axis=0)
            # result = np.min(np.sum(np.abs(target_aux_1), axis=0), axis=0)
            self.target_table[i, :] = result
            self.vp.copyto(dst=target_aux_1, src=target_aux_0)

        # create and store contour image in memory
        self.target_cv2 = create_monochromatic_image(self.image_width, self.image_height, device='cpu')
        self.target_cv2 = draw_contours(self.target_cv2, contours, copy=False)
        self.target_pil = Image.fromarray(self.target_cv2)
        if self.device == 'gpu':
            self.target_cv2 = cp.array(self.target_cv2)

    def get_target_image(self) -> Image:
        return self.target_pil

    def list_to_chunks(self, individual):
        """
        A "reversed" version of the usual list_to_chunks, used for speeding up
        the calculation of the length of the lines.
        """
        length = len(individual)
        for chunk in range(0, length // 2, self.chunk_size // 2):
            start = (int(individual[chunk] * self.image_width), int(individual[chunk+1] * self.image_height))
            end = (int(individual[length-2-chunk] * self.image_width), int(individual[length-1-chunk] * self.image_height))
            yield start, end

    def standardize_individual(self, individual, check_repr=False) -> tuple[np.ndarray, np.ndarray]:
        # reshape and rescale
        individual = individual if self.device == 'cpu' else cp.asarray(individual)
        reshaped = self.vp.reshape(individual, (self.num_points, 2))
        r0, r1 = reshaped[:, 0], reshaped[:, 1]
        r0 *= self.image_width
        r1 *= self.image_height
        # now create image for MSE
        img = create_monochromatic_image(self.image_width, self.image_height, device='cpu')
        for start, end in self.list_to_chunks(individual):
            img = cv2.line(img, start, end, color=0)
        return img, reshaped.astype(dtype=self.vp.int32).T  # first numpy/cupy image, then array for distances

    def _core_get_difference(self, individual: TArray, index: int = 0):
        individual_img, individual_points = self.standardize_individual(individual)
        aux = self.target_table[individual_points[1], individual_points[0]]
        p, q = self.vp.sum(aux), np.sum(np.abs(cv2.subtract(self.target_cv2, individual_img))) // 255
        individual_points_flipped = np.flip(individual_points, axis=0)
        r = np.sum(np.abs(individual_points - individual_points_flipped)) // 2  # total sum of the lines length
        self.results[index] = self.point_adherence_coeff * p + self.line_adherence_coeff * q + self.line_l1_lambda * r

    def get_difference(self, individuals: TArray):
        n_ind = len(individuals)
        self.results[:] = 0
        if self.device == 'cpu':
            for index in range(n_ind):
                self._core_get_difference(individuals[index], index)
        else:
            raise NotImplementedError
        results = self.results[:n_ind] if self.device == 'cpu' else cp.asnumpy(self.results[:n_ind])
        return results.copy()   # fixme check if this is necessary


__all__ = [
    'LinesNNPointContoursMetric',
]
