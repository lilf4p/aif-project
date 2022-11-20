# Miscellaneous utils
from __future__ import annotations
from .types import *


def timeit(msg: str):
    def wrapper(func):
        def new_f(*args, **kwargs):
            crt = perf_counter()
            result = func(*args, **kwargs)
            crt = perf_counter() - crt
            print(f'Elapsed time for {msg}: {crt} seconds')
            return result
        return new_f
    return wrapper


# noinspection PyUnresolvedReferences
def pil_to_cv2(pil_image: Image, start_mode='RGB', end_mode='BGR'):
    """
    Converts the given Pillow image to CV2 format. If both modes are None, returns
    an array from given pil image.
    :param pil_image: Pillow image to convert.
    :param start_mode: Pillow image mode. Possible modes are 'RGB', 'L', None.
    :param end_mode: Target CV2 image mode. Possible modes are 'BGR', 'L', None.
    """
    arr = np.array(pil_image)
    if start_mode is None and end_mode is None:
        return arr
    if start_mode == 'RGB':
        if end_mode == 'BGR':
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif end_mode == 'L':
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unrecognized end mode '{end_mode}' coupled with start mode '{start_mode}'")
    elif start_mode == 'L':
        if end_mode == 'BGR':
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif end_mode == 'L':
            return arr  # todo sure?
        else:
            raise ValueError(f"Unrecognized end mode '{end_mode}' coupled with start mode '{start_mode}'")
    else:
        raise ValueError(f"Unrecognized start mode '{start_mode}'")


# todo check if it alters cv2_image!
# noinspection PyUnresolvedReferences
def cv2_to_pil(cv2_image, start_mode='BGR', end_mode='RGB'):
    """
    Converts given CV2 image (array) into a Pillow one. If both start_mode and end_mode are None,
    it returns a Pillow image directly from the given array.
    :param cv2_image: Image in CV2 format to convert.
    :param start_mode: Starting image format for conversion. Possible values are 'BGR', 'L'.
    :param end_mode: Target image format after conversion. Possible values are 'RGB', 'L'.
    """
    if start_mode == 'BGR':
        if end_mode == 'RGB':
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        elif end_mode == 'L':
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unrecognized end mode '{end_mode}' coupled with start mode '{start_mode}'")
    elif start_mode == 'L':
        if end_mode == 'RGB':
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        elif end_mode != 'L':
            raise ValueError(f"Unrecognized end mode '{end_mode}' coupled with start mode '{start_mode}'")
    elif (start_mode is not None) or (end_mode is not None):
        raise ValueError(f"Unrecognized modes couple ('{start_mode}', '{end_mode}')")
    pil_image = Image.fromarray(cv2_image)
    return pil_image


# noinspection PyUnresolvedReferences
def find_contours(cv2_image, t_lower: TReal, t_upper: TReal, cv2_retr=cv2.RETR_EXTERNAL,
                  cv2_chain_approx=cv2.CHAIN_APPROX_NONE) -> tuple:
    """
    Applies canny edges detector to find contours of a given image.
    :param cv2_image: Image to convert.
    :param t_lower: Lower threshold values for cv2.Canny().
    :param t_upper: Upper threshold values for cv2.Canny().
    :param cv2_retr: Parameter for cv2.findContours().
    :param cv2_chain_approx: Parameter for cv2.findContours().
    :return: A tuple consisting of the image obtained by cv2.Canny, the list of contours point
    in the original image and contours hierarchy.
    """
    edged = cv2.Canny(cv2_image, t_lower, t_upper)
    contours, hierarchy = cv2.findContours(edged, cv2_retr, cv2_chain_approx)
    return edged, contours, hierarchy


def create_monochromatic_image(width: int, height: int, color: int | tuple[int] = 255,
                               mode='gray', device='cpu'):
    """
    Creates a monochromatic CV2 image.
    :param width: Width of the image.
    :param height: Height of the image.
    :param color: Color of the image. When operating in grayscale mode, this should be
    an integer in range 0-255, otherwise if mode is BGR, it should be a tuple.
    Defaults to white in grayscale (255).
    :param mode: Mode of the image: 'gray' for grayscale, 'bgr' for BGR.
    :param device: Device in which the image should be created, either 'cpu' or 'gpu'
    for the first available CUDA gpu. Defaults to 'cpu'.
    """
    # Set numerical framework
    vp = np if device == 'cpu' else (cp if device == 'gpu' else None)
    if vp is None:
        raise ValueError(f"Unknown device '{device}'")

    if mode == 'gray':
        img = vp.zeros(shape=(height, width))#, dtype=vp.uint8)
        if not isinstance(color, int) or not (0 <= color <= 255):
            raise TypeError(f"'color' should be an integer in the range [0, 255]; got {color}")
    elif mode == 'bgr':
        img = vp.zeros(shape=(height, width, 3))#, dtype=vp.uint8)
        if not isinstance(color, tuple) or len(color) != 3 or not all([0 <= v <= 255 for v in color]):
            raise TypeError(f"'color' should be a tuple of 3 elements in the range [0, 255]; got {color}")
    else:
        raise ValueError(f"Unrecognized mode '{mode}'")
    img[:, :] = color
    return img


# todo operates in grayscale
# noinspection PyUnresolvedReferences
def draw_contours(image, contours, num_contours: int = -1,
                   color: int = 0, width: int = 1, copy: bool = True):
    image_copy = image.copy() if copy else image
    cv2.drawContours(image_copy, contours, num_contours, color, width)
    return image_copy


def bresenham(start, end, as_list=False, as_tuple=False, device='cpu') -> list | np.ndarray | cp.ndarray | tuple[list, list]:
    """
    Bresenham's Line Generation Algorithm.
    Credits: https://github.com/daQuincy/Bresenham-Algorithm (slightly modified).
    """
    vp = np if device == 'cpu' else cp
    # step 1 get end-points of line
    (x0, y0) = start
    (x1, y1) = end

    # step 2 calculate difference
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    m = dy / dx if dx != 0 else None

    line_pixel = [(x0, y0)]

    # step 3 if m is None then the two points are on the same x
    if m is None:
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(y0, y1 + 1):
            line_pixel.append((x0, y))
    else:
        # step 4 perform test to check if pk < 0
        flag = True

        step = 1
        if x0 > x1 or y0 > y1:
            step = -1

        mm = False
        if m < 1:
            x0, x1, y0, y1 = y0, y1, x0, x1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            mm = True

        p0 = 2 * dx - dy
        x = x0
        y = y0

        p = p0  # not necessary since flag is always True at the start, but disables warning from IDE
        for i in range(abs(y1 - y0)):
            if flag:
                x_previous = x0
                p_previous = p0
                p = p0
                flag = False
            else:
                x_previous = x
                p_previous = p

            if p >= 0:
                x = x + step

            p = p_previous + 2 * dx - 2 * dy * (abs(x - x_previous))
            y = y + 1

            if mm:
                line_pixel.append((y, x))
            else:
                line_pixel.append((x, y))

    return line_pixel if as_list else vp.array(line_pixel)


def bresenham_tuple(start, end, device='cpu') -> tuple[list, list]:
    """
    Bresenham's Line Generation Algorithm.
    Credits: https://github.com/daQuincy/Bresenham-Algorithm (slightly modified).
    """
    vp = np if device == 'cpu' else cp
    # step 1 get end-points of line
    (x0, y0) = start
    (x1, y1) = end

    # step 2 calculate difference
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    m = dy / dx if dx != 0 else None

    line_pixel = [[x0], [y0]]

    # step 3 if m is None then the two points are on the same x
    if m is None:
        if y0 > y1:
            y0, y1 = y1, y0
        for y in range(y0, y1 + 1):
            line_pixel[0].append(x0)
            line_pixel[1].append(y)
    else:
        # step 4 perform test to check if pk < 0
        flag = True

        step = 1
        if x0 > x1 or y0 > y1:
            step = -1

        mm = False
        if m < 1:
            x0, x1, y0, y1 = y0, y1, x0, x1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            mm = True

        p0 = 2 * dx - dy
        x = x0
        y = y0

        p = p0  # not necessary since flag is always True at the start, but disables warning from IDE
        for i in range(abs(y1 - y0)):
            if flag:
                x_previous = x0
                p_previous = p0
                p = p0
                flag = False
            else:
                x_previous = x
                p_previous = p

            if p >= 0:
                x = x + step

            p = p_previous + 2 * dx - 2 * dy * (abs(x - x_previous))
            y = y + 1

            if mm:
                line_pixel[0].append(y)
                line_pixel[1].append(x)
            else:
                line_pixel[0].append(x)
                line_pixel[1].append(y)
    return line_pixel[0], line_pixel[1]


def naive_line(r0, c0, r1, c1):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = naive_line(c0, r0, c1, r1)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return naive_line(r1, c1, r0, c0)

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * (r1 - r0) / (c1 - c0) + (c1 * r0 - c0 * r1) / (c1 - c0)

    valbot = np.floor(y) - y + 1
    valtop = y - np.floor(y)

    return (np.concatenate((np.floor(y), np.floor(y) + 1)).astype(int), np.concatenate((x, x)).astype(int),
            np.concatenate((valbot, valtop)))


# Some Numba Functions (to rework)
@nb.njit
def _filter_out_nb(arr, out, cond_nb):
    j = 0
    for i in range(arr.size):
        if cond_nb(arr[i]):
            out[j] = arr[i]
            j += 1
    return j


def filter_resize_xnb(arr, cond_nb):
    result = np.empty_like(arr)
    j = _filter_out_nb(arr, result, cond_nb)
    result.resize(j, refcheck=False)  # unsupported in NoPython mode
    return result


def common_test_part(experiment, gen_step, other_callback_args, logger):
    experiment.setup()
    # experiment.plot_individual_sample(difference=False, eval_fitness=True)
    # Enable below for checking correct fitness for target
    """
    target_individual = experiment.metric.get_target_as_individual()
    print(f"Target individual: {target_individual}")
    print(f"Its fitness is: {experiment.metric.get_difference(target_individual)}")
    """
    callback_args = other_callback_args if other_callback_args is not None else {}
    callback_args['gen_step'] = gen_step
    callbacks = {
        experiment.save_image: callback_args,
    }
    if logger is not None:
        callbacks[logger] = {}
    experiment.run(show=True, callbacks=callbacks)


__all__ = [
    'timeit',
    'pil_to_cv2',
    'cv2_to_pil',
    'find_contours',
    'create_monochromatic_image',
    'draw_contours',
    'bresenham',
    'bresenham_tuple',
    'filter_resize_xnb',
    'common_test_part',
]
