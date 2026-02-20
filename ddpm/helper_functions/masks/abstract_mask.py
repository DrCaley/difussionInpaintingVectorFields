"""Abstract base class for inpainting mask generators.

Mask convention
---------------
``1.0`` = missing / to-inpaint,  ``0.0`` = known / observed.

See ``ddpm.protocols.MaskGeneratorProtocol`` for the formal contract.
"""

from abc import ABC, abstractmethod


class MaskGenerator(ABC):
    """Base class for all mask generators.

    Concrete implementations live in ``ddpm/helper_functions/masks/``
    and are auto-discovered by ``masks/__init__.py``.

    Convention: ``1.0`` = missing, ``0.0`` = known.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_mask(self, image_shape=None):
        """Return a (1, H, W) or (H, W) binary mask.

        Values: 1.0 = missing, 0.0 = known.
        """
        pass

    @abstractmethod
    def __str__(self):
        """Human-readable name for logging/plotting."""
        pass

    @abstractmethod
    def get_num_lines(self):
        """Number of observation paths/lines (for path-based masks)."""
        pass
        return 0