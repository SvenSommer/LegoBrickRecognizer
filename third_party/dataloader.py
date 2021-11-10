import cv2
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd


def make_weights_for_balanced_classes(dataset_table, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''

    count = [0] * nclasses

    for item_idx in range(len(dataset_table)):
        count[dataset_table.loc[item_idx, 'brick_class']] += 1  # item is (img-data, label-id)

    weight_per_class = [0.] * nclasses

    N = float(sum(count))  # total number of images

    for i in range(nclasses):
        d = float(count[i])
        if count[i] == 0:
            d = 1
        weight_per_class[i] = N / d

    weight = [0] * len(dataset_table)

    for item_idx in range(len(dataset_table)):
        val = dataset_table.loc[item_idx, 'brick_class']
        weight[item_idx] = weight_per_class[val]

    return weight


def create_square_crop_by_detection(
        frame: np.ndarray,
        box: list,
        return_shifts: bool = False,
        zero_pad: bool = True):
    """
    Rebuild detection box to square shape
    Args:
        frame: rgb image in np.uint8 format
        box: list with follow structure: [x1, y1, x2, y2]
        return_shifts: if set True then function return tuple of image crop
           and (x, y) tuple of shift coordinates
        zero_pad: pad result image by zeros values

    Returns:
        Image crop by box with square shape or tuple of crop and shifted coords
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w // 2
    cy = box[1] + h // 2
    radius = max(w, h) // 2
    exist_box = []
    pads = []

    # y top
    if cy - radius >= 0:
        exist_box.append(cy - radius)
        pads.append(0)
    else:
        exist_box.append(0)
        pads.append(-(cy - radius))

    # y bottom
    if cy + radius >= frame.shape[0]:
        exist_box.append(frame.shape[0] - 1)
        pads.append(cy + radius - frame.shape[0] + 1)
    else:
        exist_box.append(cy + radius)
        pads.append(0)
    # x left
    if cx - radius >= 0:
        exist_box.append(cx - radius)
        pads.append(0)

    else:
        exist_box.append(0)
        pads.append(-(cx - radius))

    # x right
    if cx + radius >= frame.shape[1]:
        exist_box.append(frame.shape[1] - 1)
        pads.append(cx + radius - frame.shape[1] + 1)
    else:
        exist_box.append(cx + radius)
        pads.append(0)

    exist_crop = frame[
                 exist_box[0]:exist_box[1],
                 exist_box[2]:exist_box[3]
                 ].copy()

    croped = np.pad(
        exist_crop,
        (
            (pads[0], pads[1]),
            (pads[2], pads[3]),
            (0, 0)
        ),
        'reflect' if not zero_pad else 'constant',
        constant_values=0
    )

    if not return_shifts:
        return croped

    shift_x = exist_box[2] - pads[2]
    shift_y = exist_box[0] - pads[0]

    return croped, (shift_x, shift_y)


class BricksDataloader(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str, shape: tuple,
                 augmentations: bool,
                 dataset_table_file: str, classes_names_file: str):
        print('Loading...')
        images_folders = [
            os.path.join(root_path, p)
            for p in os.listdir(root_path)
        ]

        images_folders.sort()

        self.classes_names = []

        if not os.path.isfile(classes_names_file):
            for folder_name in images_folders:
                cls_name = os.path.splitext(
                    os.path.basename(folder_name)
                )[0]
                self.classes_names.append(cls_name)

            self._write_classes(classes_names_file)
        else:
            with open(classes_names_file, 'r') as f:
                self.classes_names = [line.rstrip() for line in f]

        self.num_classes = len(self.classes_names)

        if not os.path.isfile(dataset_table_file):
            classes_are_updated = False
            print('Create file: {}'.format(dataset_table_file))
            with open(dataset_table_file, 'w') as f:
                f.write('img_path,brick_class\n')
                for folder_path in tqdm(images_folders):
                    cls_name = os.path.splitext(
                        os.path.basename(folder_path)
                    )[0]
                    if cls_name not in self.classes_names:
                        self.classes_names.append(cls_name)
                        classes_are_updated = True

                    cls_i = self.classes_names.index(cls_name)

                    for image_name in os.listdir(folder_path):
                        f.write(
                            '{}, {}\n'.format(
                                os.path.join(folder_path, image_name),
                                cls_i
                            )
                        )

                if classes_are_updated:
                    self._write_classes(classes_names_file)

        self.images_dataframe = pd.read_csv(dataset_table_file, sep=',')
        self.samples_count = len(self.images_dataframe)

        self.shape = shape

        self.preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.shape, interpolation=2),
                torchvision.transforms.ToTensor()
            ]
        )

        self.augmentations = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(),
                torchvision.transforms.RandomAffine(
                    15, translate=None,
                    scale=None, shear=None,
                    resample=False, fillcolor=255
                ),
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.1, p=0.5,
                    interpolation=Image.NEAREST, fill=255
                ),
                torchvision.transforms.Resize(
                    self.shape,
                    interpolation=Image.NEAREST
                ),
                torchvision.transforms.ToTensor()
            ]
        ) if augmentations else None

    def _write_classes(self, target_file_path):
        with open(target_file_path, 'w') as f:
            for cls_name in self.classes_names:
                f.write('{}\n'.format(cls_name))

    def updated_classes(self, target_instance):
        """
        Update classes info with target BricksDataloader instance
        Args:
            target_instance: BricksDataloader
        """
        self.classes_names = target_instance.classes_names
        self.num_classes = target_instance.num_classes

    def __len__(self):
        return self.samples_count

    def apply_augmentations(self, img):
        if self.augmentations is not None:
            return torch.clamp(self.augmentations(img), 0, 1)
        return self.preprocessing(img)

    def get_classes_sampler(self):
        weights = make_weights_for_balanced_classes(self.images_dataframe,
                                                    self.num_classes)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))
        return sampler

    def __getitem__(self, idx):
        img_path = self.images_dataframe.loc[idx, 'img_path']
        brick_class = self.images_dataframe.loc[idx, 'brick_class']
        image = np.array(Image.open(img_path).convert('RGB'))
        image = create_square_crop_by_detection(
            image,
            [0, 0, image.shape[1], image.shape[0]]
        )
        image = Image.fromarray(image)

        return self.apply_augmentations(image), brick_class
