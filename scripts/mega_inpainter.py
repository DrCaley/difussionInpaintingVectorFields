#############################
# BIGGER AND BETTER THAN EVER
#############################
import logging
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from data_prep.data_initializer import DDInitializer
from ddpm.testing.model_inpainter import ModelInpainter
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator
from ddpm.helper_functions.death_messages import get_death_message

print("================================================\n")
print("JARVIS: hello there, here are the models we are going to be testing:")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


directory = Path("./ddpm/trained_models") # double dot for pycharm, one dot for terminal
destination_directory = Path("./ddpm/tested_models")
destination_directory.mkdir(parents=True, exist_ok=True)
config_files = []
models = []
model_dirs = []
names = []

def move_finished_model(model_dir: Path):
    try:
        # Check if the source directory exists
        if model_dir.is_dir():
            # Move the directory
            model_dir.rename(destination_directory / model_dir.name)
            print(f"Directory '{model_dir}' moved to '{destination_directory}' successfully.")
        else:
            print(f"Source directory '{model_dir}' does not exist.")
    except FileExistsError:
        print(f"Destination '{destination_directory}' already exists and is not empty or is a file.")
    except Exception as e:
        print(f"An error occurred: {e}")

for entry in directory.iterdir():
    if entry.is_dir():
        model_dirs.append(entry)
        config_file_path = entry.absolute() / "config_used.yaml"
        if config_file_path.exists():
            config_files.append(entry.absolute() / "config_used.yaml")
            models.append(entry.absolute() / "ddpm_ocean_model_best_checkpoint.pt")
            names.append(entry.stem)
            print("- " + entry.stem)
        else:
            print(f"config file missing for {entry.stem}, skipping model.")

print("================================================")

models_bar = tqdm(models, colour="magenta", desc="👁️ models", leave=False)

for i, model_path in enumerate(models_bar):
    try:
        logging.info(f"Loading model {models[i]}")
        mi = ModelInpainter(config_path=config_files[i], model_file=models[i])
        mi.set_model_name(names[i])

        for percentage in np.linspace(1, 0, 10):
            for _ in range(10):
                mi.add_mask(CoverageMaskGenerator(percentage))

        mi.visualize_images()
        mi.find_coverage()
        mi.begin_inpainting()
        DDInitializer.reset_instance()
        move_finished_model(model_dirs[i])
    except Exception as e:
        logging.error("🚨 Oops! Something went wrong during inpatinting.")
        logging.error(f"💥 Error: {str(e)}")
        logging.error(get_death_message())
        logging.error(f"Inpainting {names[i]} crashed. Check the logs or ask your local neighborhood AI expert 🧠.")
        continue
