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
    aux = np.zeros((tot_contours, 2), dtype=np.intp)  # aux[i] is the i-th point in the contour
    i = 0
    for contour in contours_list:
        # contour = contour if self.device == 'cpu' else self.vp.array(contour)
        k = len(contour)
        aux[i:i + k, :] = contour
        i += k
    # now aux contains a column with all the widths of the target points and another one with the heights
    return aux, pil_image, cv2_image, contours


@timeit()
def build_distance_table(image_height, image_width, num_targets, aux, device='cpu'):
    target_aux_0 = np.zeros((2, num_targets, image_width), dtype=np.intp)
    target_aux_1 = np.zeros((2, num_targets, image_width), dtype=np.intp)
    # target_aux_0[0]/[1] contains the image widths and heights for all the target points for all
    # the x coordinates
    for j in range(image_width):
        target_aux_0[0, :, j] = aux[:, 0]
        target_aux_0[1, :, j] = aux[:, 1]
        target_aux_1[0, :, j] = aux[:, 0]
        target_aux_1[1, :, j] = aux[:, 1]

    # Build table of distances
    target_table = np.full((image_height, image_width), -1, dtype=np.intp)
    # Ciclo sull'altezza prima e poi sulla larghezza, quindi per l'altezza posso sottrarre la stessa
    # quantità da tutto target_aux_1[1], mentre per l'ampiezza devo sottrarre ogni volta l'ampiezza
    # corrispondente (sto facendo il calcolo per i punti (i<costante>, j=0,...,image_width-1)
    for i in range(image_height):
        target_aux_1[1] -= i
        for j in range(image_width):
            target_aux_1[0, :, j] -= j
        target_aux_1 = np.abs(target_aux_1)
        target_aux_2 = np.sum(target_aux_1, axis=0)
        result = np.min(target_aux_2, axis=0)
        target_table[i, :] = result
        np.copyto(dst=target_aux_1, src=target_aux_0)
    return target_table


__all__ = [
    'extract_contours',
    'build_distance_table',
]
