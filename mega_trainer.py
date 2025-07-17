import logging
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from ddpm.helper_functions.death_messages import get_death_message
from ddpm.training.xl_ocean_trainer import TrainOceanXL
from data_prep.data_initializer import DDInitializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESET = '\033[0m'
GREEN = '\033[32m'

directory = Path("./models_to_train")
models_to_train = []

print("JARVIS: hello there, here are the yaml files that we are going to be running-")
for entry in directory.iterdir():
    if entry.suffix == ".yaml":
        print(entry.name)
        models_to_train.append(entry)

if len(models_to_train) == 0:
    print(f"no models in {GREEN}models_to_train{RESET}")

for file in models_to_train:
    try:
        print(f"Currently training:{GREEN} {file.name}{RESET}")
        trainer = TrainOceanXL(config_path=file.absolute())
        trainer.train()
        DDInitializer.reset_instance()
    except Exception as e:
        logging.error("🚨 Oops! Something went wrong during training.")
        logging.error(f"💥 Error: {str(e)}")
        logging.error(get_death_message())
        logging.error("Training crashed. Check the logs or ask your local neighborhood AI expert 🧠.")
