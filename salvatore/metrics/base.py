# Base class for image metrics for evaluating image individuals
from __future__ import annotations
from salvatore.utils.types import *


class ImageMetric:
    """
    Base class for an image metric to be used in a Genetic Algorithm.
    It specifies the accepted chromosome formats and provides basic
    plotting utilities and calculates the cost. # todo completare!
    """
    def __init__(self, image_path: str):
        self.image_path = image_path    # We cannot even be interested in storing opened image in subclasses
        self.standardize_target()

    @abstractmethod
    def standardize_target(self):
        """
        Given the image path, standardizes the representation of the target
        for metric calculation.
        """
        pass

    @abstractmethod
    def get_target_image(self) -> Image:
        """
        Reconstruct image from target representation.
        """
        pass

    @abstractmethod
    def get_individual_image(self, individual) -> Image:
        """
        Reconstruct image from the given individual.
        """
        pass

    @abstractmethod
    def check_individual_repr(self, individual) -> TBoolStr:
        """
        Checks if individual representation is valid for this metric.
        """
        pass

    @abstractmethod
    def standardize_individual(self, individual, check_repr=False):
        """
        Converts individual into a representation that can be used for calculating difference.
        Base implementation that returns the individual itself, to be extended in subclasses.
        """
        if check_repr:
            result, msg = self.check_individual_repr(individual)
            if not result:
                raise ValueError(f"Check failed for individual {individual}: {msg}")
        return individual

    @abstractmethod
    def get_difference(self, individual):
        """
        Returns value of the metric w.r.t. the given individual.
        """
        pass


__all__ = [
    'ImageMetric',
]
