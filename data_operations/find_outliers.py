from argparse import ArgumentParser, Namespace
import numpy as np
from tqdm import tqdm
import os
from shutil import copyfile
from PIL import Image
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '../bricks_classification_server/'
    )
)

from networks import StandardClassification


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Find outliers'
    )
    parser.add_argument(
        '-m', '--model', required=True, type=str,
        help='Path to traced model file'
    )
    parser.add_argument(
        '-с', '--classes', required=True, type=str,
        help='Path to file with classes names'
    )
    parser.add_argument(
        '-i', '--input', required=True, type=str,
        help='Path to input folder'
    )
    parser.add_argument(
        '-o', '--output', required=True, type=str,
        help='Path to result folder'
    )
    return parser.parse_args()


def prepare_paths(input_folder_path: str) -> list:
    res = []
    for filename in os.listdir(input_folder_path):
        filepath = os.path.join(
            input_folder_path,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        res.append(filepath)

    return res


def save_outliers(res_folder_paths: list,
                  classes_names_and_confidences:list,
                  save_path: str):
    for sample_idx, sample_path in enumerate(res_folder_paths):
        save_sample_path = os.path.join(
            save_path,
            classes_names_and_confidences[sample_idx][0],
            '{:.2f}_'.format(
                classes_names_and_confidences[sample_idx][1]
            ) + os.path.basename(sample_path)
        )

        os.makedirs(
            os.path.join(
                save_path,
                classes_names_and_confidences[sample_idx][0]
            ),
            exist_ok=True
        )

        copyfile(sample_path, save_sample_path)


def find_outliers(
        model_path: str, classes_names_path: str, input_paths: list) -> tuple:
    cls_model = StandardClassification(model_path, classes_names_path)

    pred_names = []
    pred_vectors = []

    for input_path in tqdm(input_paths):
        out_vec, pred_cls_name = cls_model.get_prediction(
            Image.open(input_path))

        pred_names.append(pred_cls_name)
        pred_vectors.append(out_vec)

    pred_vectors = np.array(pred_vectors)

    outlier_paths = []
    outlier_classes_names = []
    outlier_confidences = []

    for class_index, class_name in enumerate(cls_model.classes_names):
        target_vectors = pred_vectors[
            pred_vectors.argmax(axis=1) == class_index][:, class_index]

        target_paths = []
        target_names = []
        for _i, am in enumerate(pred_vectors.argmax(axis=1)):
            if am == class_index:
                target_paths.append(input_paths[_i])
                target_names.append(pred_names[_i])

        normed_values = (target_vectors - target_vectors.mean()
                         ) / (target_vectors.std() + 1E-5)

        for sample_index, v in enumerate(normed_values):
            if v > 1.5:
                outlier_paths.append(target_paths[sample_index])
                outlier_confidences.append(1)
                outlier_classes_names.append(target_names[sample_index])

    return outlier_paths, outlier_classes_names, outlier_confidences


if __name__ == '__main__':
    args = parse_args()

    sample_paths = prepare_paths(args.input)
    outliers_p, outliers_n, outliers_c = find_outliers(
        args.model, args.classes, sample_paths)
    save_outliers(outliers_p, list(zip(outliers_n, outliers_c)), args.output)
