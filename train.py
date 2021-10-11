#!/usr/bin/python
import time
import os
from argparse import ArgumentParser, Namespace
from utils.image_mover import ImageMover
from utils.database_connector import DatabaseConnector
from utils.color_info import ColorInfo, Color
from third_party.train_classificator import CustomTrainingPipeline
import yaml


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
    parser.add_argument(
        '--config', type=str, default='configuration/lego_brick_recognizer_config.yaml',
        required=False, help='Path to configuration file.'
    )
    parser.add_argument(
        '--prod', action='store_true',
        help='Sets queries to production Database'
    )

    return parser.parse_args()


args = parse_args()
with open(args.config, 'r') as conf_f:
    config_dict = yaml.safe_load(conf_f)

# Connect to Database
if args.prod:
    db_connector = DatabaseConnector(config_dict['DATABASE_DEBUG'])
else:
    db_connector = DatabaseConnector(config_dict['DATABASE_PROD'])

# Color Request
colorinfo = ColorInfo(db_connector.get_cursor())
colors = colorinfo.get_colors()
for color in colors:
    print(color)

# Initialize Utils to copy images
image_mover = ImageMover(db_connector.get_cursor())

# CREATION of images to train on: Gets the labeled files from the database and moves them into the destination_folder
if not args.skip_creation:
    image_mover.move_images(args.dir)
    if not os.path.exists(args.dir):
        print("ERROR: working Folder '{}' not existing".format(args.dir))
        quit()
    print("INFO: Splitting images in training and validation set")
    # SPLITTING of the dataset into training an validation set
    classes_count = image_mover.split_train_test_dataset(args.dir)
else:
    if not os.path.exists(args.dir):
        print("ERROR: working Folder '{}' not existing".format(args.dir))
        quit()
    classes_count = len(next(os.walk(args.dir))[1])
print("INFO: Found '{}' classes".format(classes_count))

# TRAIN
# classifier = LitBrickClassifier(classes_count)
print("INFO: Started training with {} epochs on {} gpu(s).".format(args.epochs, args.gpus_count))
# classifier.trainLitBrickClassifier(args.dir, args.epochs, args.gpus_count)

classifier = CustomTrainingPipeline(
    train_data_path=os.path.join(args.dir, 'partno/'),
    val_data_path=os.path.join(args.dir, 'partno_val/'),
    experiment_folder=os.path.join(args.dir, 'experiments/'),
    stop_criteria=1E-5
)
classifier.fit()
