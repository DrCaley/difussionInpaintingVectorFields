import importlib
import inspect
from pathlib import Path

from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

__all__ = []
globals_ns = globals()

# Dynamically import all classes from other .py files
module_dir = Path(__file__).parent
mask_files = [f.stem for f in module_dir.glob("*.py") if f.name not in ("__init__.py", "base.py")]

for module_name in mask_files:
    module = importlib.import_module(f".{module_name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, MaskGenerator) and obj is not MaskGenerator:
            globals_ns[name] = obj
            __all__.append(name)
