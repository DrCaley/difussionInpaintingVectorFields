import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ddpm.Testing.model_inpainter import ModelInpainter
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator


def main():
    mi = ModelInpainter()
    mi.load_models_from_yaml()
    mi.add_mask(RobotPathGenerator())
    mi.visualize_images()
    mi.begin_inpainting()


if __name__ == "__main__":
    main()
