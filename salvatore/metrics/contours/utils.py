"""
Common utility functions for the metrics.
"""
from __future__ import annotations
from salvatore.utils import *


def extract_contours(image: np.ndarray | str, canny_low, canny_high, device='cpu'):
    pil_image = Image.open(image) if isinstance(image, str) else image
    cv2_image = pil_to_cv2(pil_image)
    image_width, image_height = cv2_image.shape[1], cv2_image.shape[0]

    # find contours
    _, contours, _ = find_contours(cv2_image, canny_low, canny_high)

    # Transform contours from numpy arrays to tuples
    tot_contours = 0
    contours_list = []
    for contour in contours:
        fldim = np.prod(contour.shape)
        contour = np.reshape(contour, (fldim // 2, 2))
        tot_contours += len(contour)
        contours_list.append(contour)
    # objective tensor: (2, num_targets, num_points)
    num_targets = tot_contours
    aux = np.zeros((tot_contours, 2))
    i = 0
    for contour in contours_list:
        # contour = contour if self.device == 'cpu' else self.vp.array(contour)
        k = len(contour)
        aux[i:i + k, :] = contour
        i += k
    # now aux contains a column with all the widths of the target points and another one with the heights
    # and it will be transferred to gpu if specified in constructor
    # aux = aux.astype(np.int32)  # todo pass integer type as parameter!
    aux = aux if device == 'cpu' else cp.asarray(aux)
    return aux, pil_image, cv2_image, contours


def build_distance_table(image_height, image_width, num_targets, aux, dtype=np.int32, device='cpu'):
    vp = np if device == 'cpu' else cp
    target_aux_0 = vp.zeros((2, num_targets, image_width))
    target_aux_1 = vp.zeros((2, num_targets, image_width))
    for j in range(image_width):
        target_aux_0[0, :, j] = aux[:, 0]
        target_aux_0[1, :, j] = aux[:, 1]
        target_aux_1[0, :, j] = aux[:, 0]
        target_aux_1[1, :, j] = aux[:, 1]

    # Build table of distances
    target_table = vp.full((image_height, image_width), -1, dtype=dtype)
    for i in range(image_height):
        target_aux_1[1] -= i
        for j in range(image_width):  # todo questo si può modificare con un array (width,) creato appositamente!
            target_aux_1[0, :, j] -= j
        target_aux_1 = vp.abs(target_aux_1)
        target_aux_2 = vp.sum(target_aux_1, axis=0)
        result = vp.min(target_aux_2, axis=0)
        target_table[i, :] = result
        vp.copyto(dst=target_aux_1, src=target_aux_0)
    return target_table


__all__ = [
    'extract_contours',
    'build_distance_table',
]
