import cv2
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd


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
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
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
                # torchvision.transforms.RandomGrayscale(p=0.1),
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

    def __getitem__(self, idx):
        img_path = self.images_dataframe.loc[idx, 'img_path']
        brick_class = self.images_dataframe.loc[idx, 'brick_class']
        image = Image.open(img_path)

        return self.apply_augmentations(image), brick_class
