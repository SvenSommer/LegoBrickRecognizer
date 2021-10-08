#!/usr/bin/python
import time
from argparse import ArgumentParser, Namespace
from utils.image_mover import ImageMover


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Prepares and trains a new model for brick classification')
    parser.add_argument(
        '--dir', type=str, required=False,
        default='./training_data/training_export_' + time.strftime('%Y%m%d_%H%M%S'),
        help='Directory to the folder which will contain the training files'
    )
    return parser.parse_args()


args = parse_args()
image_mover = ImageMover()

# Gets the labeled files from the database and moves them into the destination_folder
image_mover.move_images(args.dir)
