import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from ddpm.training.xl_ocean_trainer import TrainOceanXL
from data_prep.data_initializer import DDInitializer

directory = Path("./models_to_train")
yamls = []

print("JARVIS: hello there, here are the yaml files that we are going to be running-")
for entry in directory.iterdir():
    if entry.suffix == ".yaml":
        print(entry.name)
        yamls.append(entry)

for file in yamls:
    print(f"Currently training: {file.name}")
    trainer = TrainOceanXL(config_path=file.absolute())
    trainer.train()
    DDInitializer.reset_instance()
