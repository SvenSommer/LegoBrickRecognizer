from argparse import ArgumentParser, Namespace
import numpy as np
import os
from tqdm import tqdm
from shutil import copyfile


class DataSetSplitter(object):
    def __init__(self,
                 input_folder: str,
                 output_folder: str,
                 validation_part_ratio: float):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.validation_part_ratio = validation_part_ratio

    def split(self):
        for class_name in tqdm(os.listdir(self.input_folder)):
            res_folder = os.path.join(self.output_folder, class_name)
            os.makedirs(res_folder, exist_ok=True)

            images_names = list(os.listdir(os.path.join(self.input_folder, class_name)))
            for _ in range(int(len(images_names) * self.validation_part_ratio)):
                selected_id = np.random.randint(0, len(images_names))
                inp_img_path = os.path.join(
                    self.input_folder,
                    class_name,
                    images_names[selected_id]
                )

                copyfile(
                    inp_img_path,
                    os.path.join(res_folder, images_names[selected_id])
                )

                os.remove(inp_img_path)
                del images_names[selected_id]


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Create new folder and separate data.'
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to input folder.'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to output folder.'
    )
    parser.add_argument(
        '-k', '--k', type=float, required=False, default=0.2,
        help='Validation part ratio.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DataSetSplitter(args.input,
                    args.output,
                    args.k).split()

    exit(0)