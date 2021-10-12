from argparse import ArgumentParser, Namespace
import os
from tqdm import tqdm
from shutil import copyfile


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Flatten colors_id dataset by color types.'
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to input folder.'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to output folder.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    inp_folder = args.input
    out_folder = args.output

    os.makedirs(out_folder, exist_ok=True)

    color_types = []

    for camera_type in tqdm(os.listdir(inp_folder)):
        camera_type_path = os.path.join(inp_folder, camera_type)

        for color_type in os.listdir(camera_type_path):
            if color_type not in color_types:
                color_types.append(color_type)
                os.makedirs(os.path.join(out_folder, color_type), exist_ok=True)

            for color_name in os.listdir(
                    os.path.join(camera_type_path, color_type)):
                color_folder = os.path.join(
                    camera_type_path,
                    color_type,
                    color_name
                )

                for image_name in os.listdir(color_folder):
                    inp_image_path = os.path.join(color_folder, image_name)
                    out_image_path = os.path.join(
                        out_folder,
                        color_type,
                        image_name
                    )

                    if os.path.isfile(out_image_path):
                        raise RuntimeWarning(
                            'File {} is exist'.format(out_image_path))

                    copyfile(inp_image_path, out_image_path)
