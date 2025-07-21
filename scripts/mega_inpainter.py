#############################
# BIGGER AND BETTER THAN EVER
#############################

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from data_prep.data_initializer import DDInitializer
from ddpm.testing.model_inpainter import ModelInpainter
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator

print("JARVIS: hello there, here are the models we are going to be testing:")

directory = Path("./ddpm/trained_models")
config_files = []
models = []
names = []

for entry in directory.iterdir():
    if entry.is_dir():
        config_files.append(entry.absolute() / "config_used.yaml")
        models.append(entry.absolute() / "ddpm_ocean_model_best_checkpoint.pt")
        names.append(entry.stem)
        print(entry.stem)

for i in range(len(models)):
    mi = ModelInpainter(config_path=config_files[i], model_file=models[i])
    mi.set_model_name(names[i])

    for _ in range(1):
        mi.add_mask(CoverageMaskGenerator(0.5))

    mi.visualize_images()
    mi.find_coverage()
    mi.begin_inpainting()
    DDInitializer.reset_instance()