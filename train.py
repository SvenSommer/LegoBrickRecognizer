#!/usr/bin/python
import time
import os
from argparse import ArgumentParser, Namespace
from utils.image_mover import ImageMover
from utils.database_connector import DatabaseConnector
from utils.color_info import ColorInfo, Color
from third_party.train_classificator import CustomTrainingPipeline
from data_operations.split_classification_dataset_to_train_val_folders import DataSetSplitter
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

    parser.add_argument(
        '--reduce-partno', action='store_true',
        help='Reduces the partno to its base no'
    )

    return parser.parse_args()


args = parse_args()
folder_dict = [{'train': os.path.join(args.dir, 'partno/'),
                'validation': os.path.join(args.dir, 'partno_val/'),
                'model': os.path.join(args.dir, 'partno_model/')},
               {'train': os.path.join(args.dir, 'color_id/USB'),
                'validation': os.path.join(args.dir, 'color_id_val/USB'),
                'model': os.path.join(args.dir, 'color_id_model_USB/')},
               {'train': os.path.join(args.dir, 'color_id/BRIO'),
                'validation': os.path.join(args.dir, 'color_id_val/BRIO'),
                'model': os.path.join(args.dir, 'color_id_model_BRIO/')}]

with open(args.config, 'r') as conf_f:
    config_dict = yaml.safe_load(conf_f)

# Connect to Database
if args.prod:
    db_connector = DatabaseConnector(config_dict['DATABASE_DEBUG'])
else:
    db_connector = DatabaseConnector(config_dict['DATABASE_PROD'])

# Initialize Utils to copy images
image_mover = ImageMover(db_connector.get_cursor())

# CREATION of images to train on: Gets the labeled files from the database and moves them into the destination_folder
if not args.skip_creation:
    print("INFO: reduce_partno is", args.reduce_partno)
    image_mover.create_training_dir_partno(args.dir, args.reduce_partno)
    image_mover.create_training_dir_color_id(args.dir)

    if not os.path.exists(args.dir):
        print("ERROR: working Folder '{}' not existing".format(args.dir))
        quit()
    print("INFO: Splitting images in training and validation set")
    # SPLITTING of the dataset into training an validation set

    for folder in folder_dict:
        print("Working on folder {}".format(folder['train']))
        splitter = DataSetSplitter(folder['train'], folder['validation'], 0.2).split()

if not os.path.exists(args.dir):
    print("ERROR: working Folder '{}' not existing".format(args.dir))
    quit()
classes_count = len(next(os.walk(args.dir))[1])
print("INFO: Found '{}' classes".format(classes_count))
for folder in folder_dict:
    print("INFO: Started training on {} with {} epochs on {} gpu(s).".format(folder['train'], args.epochs,
                                                                             args.gpus_count))
    # TRAIN
    classifier = CustomTrainingPipeline(
        train_data_path=folder['train'],
        val_data_path=folder['validation'],
        experiment_folder=folder['model'],
        stop_criteria=1E-5
    )
    classifier.fit()
