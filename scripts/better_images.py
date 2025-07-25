import sys
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class ImageBundle():
    def __init__(self, ddpm_image : Path, gp_image : Path, mask_image : Path, initial_image : Path):
        self.ddpm_image = ddpm_image
        self.gp_image = gp_image
        self.mask_image = mask_image
        self.initial_image = initial_image

    def __str__(self):
        return f"ImageBundle(ddpm={self.ddpm_image.name}, gp={self.gp_image.name}, mask={self.mask_image.name}, initial={self.initial_image.name})"

def draw_image(ddpm_image : Path, gp_image : Path, mask_image : Path, initial_image : Path, model_name : str):
    combo_results_path = Path(f"../combo_results/{model_name}")
    combo_results_path.mkdir(parents=True, exist_ok=True)
    file_name = mask_image.name[4:-4]

    left_img = Image.open(initial_image)
    middle_img = Image.open(mask_image)
    right_top_img = Image.open(ddpm_image)
    right_bot_img = Image.open(gp_image)

    size = (800, 374)

    gap = 30
    canvas_width = size[0] * 3 + gap * 3
    canvas_height = size[1] * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="black")

    canvas.paste(left_img, (gap, size[1] // 2 + gap // 2))
    canvas.paste(middle_img, (size[0] + gap * 4, size[1] // 3 + gap // 2))
    canvas.paste(right_top_img, (size[0] * 2 + gap * 3, 0))
    canvas.paste(right_bot_img, (size[0] * 2 + gap * 3, size[1] + gap))

    canvas.save(combo_results_path / f'{file_name}_combo.png')

if __name__ == '__main__':

    # source = input("enter directory of where all models produced results, should be a \'results\' directory:\n")
    source = "../ddpm/testing/results"
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError

    models = []

    for entry in source.iterdir():
        if entry.is_dir():
            models.append(entry.absolute() / "pt_visualizer_images/pt_predictions")

    pictures_dictionary = {}

    for model in models:
        for entry in model.iterdir():
            name = entry.name

            for prefix in ["ddpm", "gp_field", "mask", "initial"]:
                if name.startswith(prefix):
                    index = name[len(prefix)]
                    if name[len(prefix) + 1] != '_':
                        index += name[len(prefix) + 1]
                    index = int(index)

                    if index not in pictures_dictionary:
                        pictures_dictionary[index] = {}

                    pictures_dictionary[index][prefix] = entry
                    break

        bundle_dict = {}

        for key in sorted(pictures_dictionary.keys()):
            entry = pictures_dictionary[key]
            try:
                bundle = ImageBundle(
                    ddpm_image=entry["ddpm"],
                    gp_image=entry["gp_field"],
                    mask_image=entry["mask"],
                    initial_image=entry["initial"]
                    )
                bundle_dict[key] = bundle
            except KeyError as e:
                print(f"Missing image for index {key}: {e}")
                raise e

        for entry in bundle_dict.keys():
            bundle = bundle_dict[entry]
            draw_image(bundle.ddpm_image, bundle.gp_image, bundle.mask_image, bundle.initial_image, model.parent.parent.name)