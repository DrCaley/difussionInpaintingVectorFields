import sys
from pathlib import Path
from PIL import Image
import re
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class ImageBundle:
    def __init__(self, ddpm_image : Path, gp_image : Path, mask_image : Path, initial_image : Path):
        self.ddpm_image = ddpm_image
        self.gp_image = gp_image
        self.mask_image = mask_image
        self.initial_image = initial_image

class ErrorBundle:
    def __init__(self, ddpm_angular_image : Path, ddpm_mag_image : Path, ddpm_mse_image : Path, ddpm_pe_image : Path, ddpm_vector_image : Path,
                 gp_angular_image : Path, gp_mag_image : Path, gp_mse_image : Path, gp_pe_image : Path, gp_vector_image : Path):
        self.ddpm_angular_image = ddpm_angular_image
        self.ddpm_mag_image = ddpm_mag_image
        self.ddpm_mse_image = ddpm_mse_image
        self.ddpm_pe_image = ddpm_pe_image
        self.ddpm_vector_image = ddpm_vector_image

        self.gp_angular_image = gp_angular_image
        self.gp_mag_image = gp_mag_image
        self.gp_mse_image = gp_mse_image
        self.gp_pe_image = gp_pe_image
        self.gp_vector_image = gp_vector_image

def draw_fields(image_bundle: ImageBundle, model_name: str):
    combo_results_path = Path(f"../combo_results/{model_name}")
    combo_results_path.mkdir(parents=True, exist_ok=True)
    file_name = image_bundle.mask_image.name[4:-4]

    left_img = Image.open(image_bundle.initial_image)
    middle_img = Image.open(image_bundle.mask_image)
    right_top_img = Image.open(image_bundle.ddpm_image)
    right_bot_img = Image.open(image_bundle.gp_image)

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

def draw_errors(image_bundle: ErrorBundle, model_name: str):
    combo_results_path = Path(f"../combo_results/{model_name}")
    combo_results_path.mkdir(parents=True, exist_ok=True)
    file_name = image_bundle.ddpm_angular_image.name[4:-28]

    original_size = (800, 374)
    scale_factor = 0.5  # shrink to 50%
    size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    gap = 20

    canvas_width = size[0] * 5 + gap * 6
    canvas_height = size[1] * 2 + gap * 3

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="black")

    # Load and resize images
    images = [
        Image.open(image_bundle.ddpm_angular_image).resize(size),
        Image.open(image_bundle.ddpm_mag_image).resize(size),
        Image.open(image_bundle.ddpm_mse_image).resize(size),
        Image.open(image_bundle.ddpm_pe_image).resize(size),
        Image.open(image_bundle.ddpm_vector_image).resize(size),
        Image.open(image_bundle.gp_angular_image).resize(size),
        Image.open(image_bundle.gp_mag_image).resize(size),
        Image.open(image_bundle.gp_mse_image).resize(size),
        Image.open(image_bundle.gp_pe_image).resize(size),
        Image.open(image_bundle.gp_vector_image).resize(size),
    ]

    # Paste them into 2 rows of 5
    for idx, img in enumerate(images):
        row = idx // 5
        col = idx % 5
        x = gap + col * (size[0] + gap)
        y = gap + row * (size[1] + gap)
        canvas.paste(img, (x, y))

    canvas.save(combo_results_path / f'{file_name}_combo_errors.png')


if __name__ == '__main__':

    source = input("enter directory of where all models produced results, should be a \'results\' directory:\n")
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError

    models = []

    for entry in source.iterdir():
        if entry.is_dir():
            models.append(entry.absolute())


    for model in models:
        pictures_dictionary = {}
        print(model.parent.parent)
        visualizer_dir = model / "pt_visualizer_images/pt_predictions"

        for entry in visualizer_dir.iterdir():
            name = entry.name
            for prefix in ["ddpm", "gp_field", "mask", "initial"]:
                if name.startswith(prefix):
                    index = ""
                    i = 0
                    while name[len(prefix) + i] != '_':
                        index += name[len(prefix) + i]
                        i += 1
                    index = int(index)

                    match = re.search(r'num_lines_([0-9.]+).*', name)
                    if not match:
                        print(f"⚠️ Couldn't parse: {name}")
                        continue
                    num_lines = match.group(1).rstrip('.')

                    key = (index, num_lines)

                    if key not in pictures_dictionary:
                        pictures_dictionary[key] = {}

                    pictures_dictionary[key][prefix] = entry
                    break

        bundle_dict = {}

        for key in sorted(pictures_dictionary.keys()):
            entry = pictures_dictionary[key]
            try:
                bundle = ImageBundle(
                    ddpm_image=entry["ddpm"],
                    gp_image=entry["gp_field"],
                    mask_image=entry["mask"],
                    initial_image=entry["initial"],
                    )
                bundle_dict[key] = bundle
            except KeyError as e:
                print(f"Missing image for index {key}: {e}")
                raise e

        for entry in bundle_dict.keys():
            bundle = bundle_dict[entry]
            draw_fields(bundle, model.name)

    for model in models:
        pictures_dictionary = {}
        print(model.parent.parent)
        errors_dir = model / "pt_visualizer_images/pt_errors"

        for entry in errors_dir.iterdir():
            name = entry.name
            matched = False
            for prefix in ["ddpm", "gp_field"]:
                if name.startswith(prefix):
                    for err_type in ["angular", "mag", "mse", "PE", "vector"]:
                        if err_type in name:
                            match = re.search(r'^[a-z_]+(\d+)_num_lines_(\d+(?:\.\d+)?)_', name)
                            if match:
                                index = int(match.group(1))
                                num_lines = match.group(2)
                                key = (index, num_lines)

                                if key not in pictures_dictionary:
                                    pictures_dictionary[key] = {}

                                if prefix not in pictures_dictionary[key]:
                                    pictures_dictionary[key][prefix] = {}

                                pictures_dictionary[key][prefix][err_type] = entry
                                matched = True
                                break
                if matched:
                    break

        bundle_dict = {}

        for key in sorted(pictures_dictionary.keys()):
            entry = pictures_dictionary[key]
            try:
                bundle = ErrorBundle(
                    ddpm_angular_image = entry['ddpm']['angular'],
                    ddpm_mse_image = entry['ddpm']['mse'],
                    ddpm_pe_image= entry['ddpm']['PE'],
                    ddpm_mag_image= entry['ddpm']['mag'],
                    ddpm_vector_image= entry['ddpm']['vector'],
                    gp_angular_image = entry['gp_field']['angular'],
                    gp_mse_image= entry['gp_field']['mse'],
                    gp_pe_image= entry['gp_field']['PE'],
                    gp_mag_image = entry['gp_field']['mag'],
                    gp_vector_image = entry['gp_field']['vector'],
                    )
                bundle_dict[key] = bundle
            except KeyError as e:
                print(f"Missing image for index {key}: {e}")
                raise e

        for entry in bundle_dict.keys():
            bundle = bundle_dict[entry]
            draw_errors(bundle, model.name)
