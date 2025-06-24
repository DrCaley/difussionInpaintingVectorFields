from abc import ABC, abstractmethod

class MaskGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_mask(self, image_shape = None):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_num_lines(self):
        return 0