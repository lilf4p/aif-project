# Miscellaneous utils
from __future__ import annotations
import os
import re
import imageio

from .types import *


def timeit(msg: str = None):
    def wrapper(func):
        nonlocal msg
        msg = msg if msg is not None else func.__name__
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


# noinspection PyUnresolvedReferences
def draw_contours(image, contours, num_contours: int = -1,
                   color: int = 0, width: int = 1, copy: bool = True):
    image_copy = image.copy() if copy else image
    cv2.drawContours(image_copy, contours, num_contours, color, width)
    return image_copy


def common_test_part(experiment, save_image_gen_step, other_callback_args, logger, stopping_criterions=None):
    experiment.setup()
    # experiment.plot_individual_sample(difference=False, eval_fitness=True)
    # Enable below for checking correct fitness for target
    """
    target_individual = experiment.metric.get_target_as_individual()
    print(f"Target individual: {target_individual}")
    print(f"Its fitness is: {experiment.metric.get_difference(target_individual)}")
    """
    callback_args = other_callback_args if other_callback_args is not None else {}
    callback_args['gen_step'] = save_image_gen_step
    callbacks = {
        experiment.save_image: callback_args,
        experiment.save_stats: {},
    }
    stopping_criterions = {} if stopping_criterions is None else stopping_criterions
    callbacks.update(stopping_criterions)
    if logger is not None:
        callbacks[logger] = {}
        logger.set_experiment_vals(experiment)
    experiment.run(show=True, callbacks=callbacks)


def create_gif(
        dir_path: str, regex: str = r'^After [0-9]+00 generations.png',
        duration=0.25, final_duration=10, out_file_name='best_individuals.gif'
):
    """
    Creates a gif from the results of an experiment by concatenating all images
    within a given directory that matches the given regex.
    :param dir_path: Directory in which images are contained.
    :param duration: Duration of each frame of the image (except last).
    :param final_duration: Duration (in frames) of the last image.
    :param regex: String regex to use for filtering files.
    :param out_file_name: Name of the output .gif file.
    """
    def key_fun(file_name: str):
        splits = file_name.split()
        return int(splits[1])
    file_names = list(filter(
        lambda file_name: re.match(regex, file_name) is not None,
        os.listdir(dir_path)
    ))
    if len(file_names) > 0:
        file_names = sorted(file_names, key=key_fun)
        # Artificially make last frame to last for final_duration * duration seconds
        last_file_name = file_names[-1]
        for i in range(1, final_duration):
            file_names.append(last_file_name)
        # solution taken from:
        # https://stackoverflow.com/questions/41228209/making-gif-from-images-using-imageio-in-python
        with imageio.get_writer(out_file_name, mode='I', duration=duration) as writer:
            for filename in file_names:
                image = imageio.imread(filename)
                writer.append_data(image)
        writer.close()


__all__ = [
    'timeit',
    'pil_to_cv2',
    'cv2_to_pil',
    'find_contours',
    'create_monochromatic_image',
    'draw_contours',
    'common_test_part',
    'create_gif',
]
