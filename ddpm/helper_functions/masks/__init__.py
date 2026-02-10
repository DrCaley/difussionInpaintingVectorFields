import importlib
import inspect
from pathlib import Path

from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

__all__ = []
globals_ns = globals()

# Dynamically import all classes from other .py files
module_dir = Path(__file__).parent
excluded_modules = {
    "n_coverage_mask",
    "random_mask",
    "border_mask",
    "robot_ocean_mask",
    "squiggly_line",
    "random_path",
    "smile_mask",
    "no_mask",
    "better_robot_path",
}
mask_files = [
    f.stem for f in module_dir.glob("*.py")
    if f.name not in ("__init__.py", "base.py") and f.stem not in excluded_modules
]

for module_name in mask_files:
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, MaskGenerator) and obj is not MaskGenerator:
                globals_ns[name] = obj
                __all__.append(name)
    except ImportError:
        # Skip modules that can't be imported (e.g., mask_drawer requires tkinter)
        pass
