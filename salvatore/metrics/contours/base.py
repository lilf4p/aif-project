from __future__ import annotations
from salvatore.utils import *
from ..base import ImageMetric


class ContoursMetric(ImageMetric):

    def __init__(self, image_path: str, canny_low: TReal, canny_high: TReal,
                 bounds_low: TReal = 0.0, bounds_high: TReal = 1.0, device='cpu', **extra_args):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.image_width = None
        self.image_height = None
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.device = device
        super(ContoursMetric, self).__init__(image_path)


class ContoursLineMetric(ContoursMetric):

    def check_individual_repr(self, individual) -> TBoolStr:
        """
        Individual should be represented as a list with length a multiple of 4, where each
        disjoint subsequence of 4 elements from the start represent coordinates of the two
        extremes of the line.
        """
        if not isinstance(individual, Sequence):
            return False, f"Invalid type for individual: Sequence, got {type(individual)}"
        ilen = len(individual)
        if ilen % 4 != 0:
            return False, f"Invalid length for individual sequence: expected multiple of 4, got {ilen} (% 4 = {ilen % 4})"
        for i in range(ilen):
            ith = individual[i]
            if not isinstance(ith, int):
                return False, f"Invalid object at index {i}: expected int, got {type(ith)}"
            if not (self.bounds_low <= ith <= self.bounds_high):
                return False, f"Invalid value {ith} at index {i}"
        return True, None

    def list_to_chunks(self, individual):
        """
        Breaks individual into a sequence of couples of tuples
        ((start_x, start_y), (end_x, end_y)) already rescaled.
        """
        for chunk in range(0, len(individual), self.CHUNKSIZE):
            start = (int(individual[chunk] * self.image_width), int(individual[chunk+1] * self.image_height))
            end = (int(individual[chunk+2] * self.image_width), int(individual[chunk+3] * self.image_height))
            yield start, end

    def get_individual_image(self, individual) -> Image:
        image = Image.new('F', (self.image_width, self.image_height), color=255)
        draw = ImageDraw.Draw(image, 'F')
        for start, end in self.list_to_chunks(individual):
            draw.line([start, end], fill=0, width=0)
        # cleanup
        del draw
        return image


__all__ = [
    'ContoursMetric',
    'ContoursLineMetric',
]