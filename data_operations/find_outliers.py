from argparse import ArgumentParser, Namespace
import numpy as np
from tqdm import tqdm
import os
from shutil import copyfile
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor
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
    parser.add_argument(
        '-t', '--threshold', required=False, type=float, default=0.5,
        help='Filtration threshold'
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
    os.makedirs(save_path, exist_ok=True)

    for sample_idx, sample_path in enumerate(res_folder_paths):
        save_sample_path = os.path.join(
            save_path,
            # classes_names_and_confidences[sample_idx][0],
            '{:.2f}_'.format(
                classes_names_and_confidences[sample_idx][1]
            ) + os.path.basename(sample_path)
        )

        # os.makedirs(
        #     os.path.join(
        #         save_path,
        #         classes_names_and_confidences[sample_idx][0]
        #     ),
        #     exist_ok=True
        # )

        copyfile(sample_path, save_sample_path)


def find_outliers(
        model_path: str,
        classes_names_path: str,
        input_paths: list,
        threshold: float = 0.5) -> tuple:
    """
    Find outliers in folder
    Args:
        model_path: path to traced model file
        classes_names_path: path fo text file with classes names
        input_paths: List with input images paths
        threshold: filtration radius threshold, see: https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html
            the less threshold the greater filtered count

    Returns:
        List with outlier paths, outliers classes names, filtration confidences
    """
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
            pred_vectors.argmax(axis=1) == class_index]

        if len(target_vectors) < 2:
            continue

        target_paths = []
        target_names = []
        for _i, am in enumerate(pred_vectors.argmax(axis=1)):
            if am == class_index:
                target_paths.append(input_paths[_i])
                target_names.append(pred_names[_i])

        clf = LocalOutlierFactor(
            n_neighbors=max(len(target_vectors) // 10, 1),
            contamination=0.1
        )

        _ = clf.fit_predict(target_vectors)
        X_scores = clf.negative_outlier_factor_
        radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min() + 1E-5)

        # normed_values = (target_vectors - target_vectors.mean()
        #                  ) / (target_vectors.std() + 1E-5)

        for sample_index, r in enumerate(radius):
            if r - threshold > 1E-5:
                outlier_paths.append(target_paths[sample_index])
                outlier_confidences.append(r / 1)
                outlier_classes_names.append(target_names[sample_index])

    return outlier_paths, outlier_classes_names, outlier_confidences


if __name__ == '__main__':
    args = parse_args()

    sample_paths = prepare_paths(args.input)
    outliers_p, outliers_n, outliers_c = find_outliers(
        args.model, args.classes, sample_paths, args.threshold)
    save_outliers(outliers_p, list(zip(outliers_n, outliers_c)), args.output)
