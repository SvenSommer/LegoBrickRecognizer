#!/usr/bin/python
import time
import os
from argparse import ArgumentParser, Namespace
from utils.image_mover import ImageMover
from third_party.lightning_pipeline import LitBrickClassifier


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Prepares and trains a new model for brick classification')
    parser.add_argument(
        '--dir', type=str, required=False,
        default='./training_data/training_export_' + time.strftime('%Y%m%d_%H%M%S'),
        help='Directory to the folder which will contain the training files'
    )
    parser.add_argument(
        '--skip-creation', action='store_true',
        help='Skips the first step of creating a new directory with labeled data'
    )
    parser.add_argument(
        '--epochs', type=int, required=False, default=200,
        help='Epochs count for training'
    )
    parser.add_argument(
        '--gpus_count', type=int, required=False, default=1,
        help='GPU used for training'
    )

    return parser.parse_args()


args = parse_args()
image_mover = ImageMover()
classifier = LitBrickClassifier()

# CREATION of images to train on: Gets the labeled files from the database and moves them into the destination_folder
if not args.skip_creation:
    image_mover.move_images(args.dir)

if not os.path.exists(args.dir):
    print("ERROR: working Folder '{}' not existing".format(args.dir))
    quit()

# SPLITTING of the dataset into training an validation set
image_mover.split_train_test_dataset(args.dir)

# TRAIN
classifier.train(args.dir, args.epochs, args.gpus_count)
