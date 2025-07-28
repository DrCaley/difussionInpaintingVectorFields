import sys
from pathlib import Path
from PIL import Image
import re
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

class ImageBundle():
    def __init__(self, ddpm_image : Path, gp_image : Path, mask_image : Path, initial_image : Path, angular_image : Path,
                 mag_image : Path, mse_image : Path, pe_image : Path, vector_image : Path):
        self.ddpm_image = ddpm_image
        self.gp_image = gp_image
        self.mask_image = mask_image
        self.initial_image = initial_image

        self.angular_image = angular_image
        self.mag_image = mag_image
        self.mse_image = mse_image
        self.pe_image = pe_image
        self.vector_image = vector_image

def draw_image(image_bundle: ImageBundle, model_name: str):
    combo_results_path = Path(f"../combo_results/{model_name}")
    combo_results_path.mkdir(parents=True, exist_ok=True)
    file_name = image_bundle.mask_image.name[4:-4]

    # Load main images
    left_img = Image.open(image_bundle.initial_image)
    middle_img = Image.open(image_bundle.mask_image)
    right_top_img = Image.open(image_bundle.ddpm_image)
    right_bot_img = Image.open(image_bundle.gp_image)

    # Settings
    size = (1000, 400)
    gap = 30

    # Load extra images
    angular_img = Image.open(image_bundle.angular_image).resize(size)
    mag_img = Image.open(image_bundle.mag_image).resize(size)
    mse_img = Image.open(image_bundle.mse_image).resize(size)
    pe_img = Image.open(image_bundle.pe_image).resize(size)
    vector_img = Image.open(image_bundle.vector_image).resize(size)

    # Top section: 2 rows of initial/mask/ddpm/gp layout
    top_canvas_width = size[0] * 3 + gap * 4
    top_canvas_height = size[1] * 2 + gap

    # Bottom section: 2 rows of 3 and 2 images
    grid_columns = [3, 2]
    grid_rows = len(grid_columns)
    grid_img_count = 5

    bottom_grid_width = size[0] * 3 + gap * 4  # same width as top
    bottom_grid_height = size[1] * grid_rows + gap * (grid_rows + 1)

    total_canvas_height = top_canvas_height + bottom_grid_height

    # Create canvas
    canvas = Image.new("RGB", (top_canvas_width, total_canvas_height), color="black")

    # Top images
    canvas.paste(left_img, (gap, size[1] // 2 + gap // 2))
    canvas.paste(middle_img, (size[0] + gap * 2, size[1] // 3 + gap // 2))
    canvas.paste(right_top_img, (size[0] * 2 + gap * 3, 0))
    canvas.paste(right_bot_img, (size[0] * 2 + gap * 3, size[1] + gap))

    # Bottom grid images
    grid_images = [angular_img, mag_img, mse_img, pe_img, vector_img]
    y_offset = top_canvas_height

    img_index = 0
    for row, cols in enumerate(grid_columns):
        row_y = y_offset + row * (size[1] + gap) + gap
        total_row_width = cols * size[0] + (cols + 1) * gap
        x_start = (top_canvas_width - total_row_width) // 2 + gap  # center the row

        for col in range(cols):
            if img_index >= grid_img_count:
                break
            x = x_start + col * (size[0] + gap)
            canvas.paste(grid_images[img_index], (x, row_y))
            img_index += 1

    canvas.save(combo_results_path / f'{file_name}_combo.png')

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
        errors_dir = model / "pt_visualizer_images/pt_errors"

        for entry in visualizer_dir.iterdir():
            name = entry.name
            for prefix in ["ddpm", "gp_field", "mask", "initial"]:
                if name.startswith(prefix):
                    index = name[len(prefix)]
                    if name[len(prefix) + 1] != '_':
                        index += name[len(prefix) + 1]
                    index = int(index)

                    match = re.search(r'num_lines_([0-9.]+).*', name)
                    num_lines = match.group(1).rstrip('.')

                    key = (index, num_lines)

                    if key not in pictures_dictionary:
                        pictures_dictionary[key] = {}

                    pictures_dictionary[key][prefix] = entry
                    break

        for entry in errors_dir.iterdir():
            name = entry.name
            for prefix in ["angular", "mag", "mse", "PE", "vector"]:
                if prefix in name:
                    match = re.search(r'^[a-z_]+(\d+)_num_lines_(\d+(?:\.\d+)?)_', name)
                    index = int(match.group(1))
                    num_lines = match.group(2)

                    key = (index, num_lines)

                    pictures_dictionary[key][prefix] = entry

        bundle_dict = {}

        for key in sorted(pictures_dictionary.keys()):
            entry = pictures_dictionary[key]
            try:
                bundle = ImageBundle(
                    ddpm_image=entry["ddpm"],
                    gp_image=entry["gp_field"],
                    mask_image=entry["mask"],
                    initial_image=entry["initial"],
                    angular_image=entry["angular"],
                    mag_image=entry['mag'],
                    mse_image=entry["mse"],
                    pe_image=entry["PE"],
                    vector_image=entry["vector"]
                    )
                bundle_dict[key] = bundle
            except KeyError as e:
                print(f"Missing image for index {key}: {e}")
                raise e

        for entry in bundle_dict.keys():
            bundle = bundle_dict[entry]
            draw_image(bundle, model.name)