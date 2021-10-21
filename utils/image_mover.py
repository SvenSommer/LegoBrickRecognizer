#!/usr/bin/python
import pymysql
import os
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
from PIL import Image


class ImageMover:
    def __init__(self, cursor):
        self.image_counter = 0
        self.images_skipped = 0
        self.cur = cursor

    def create_training_dir_color_id(self, dest_folder):
        self.cur.execute("""SELECT path, p.color_id FROM LegoSorterDB.Partimages i 
        LEFT JOIN LegoSorterDB.Identifiedparts p ON p.id = i.part_id
        LEFT JOIN LegoSorterDB.Colors c ON p.color_id = c.color_id
        WHERE i.deleted IS NULL AND p.deleted IS NULL
        and p.color_id != 0
        and camera NOT IN ('BRIO_lower', 'BRIO', 'BRIO_center', 'unknown', 'USB')""")

        sqlresult = self.cur.fetchall()
        print("INFO: [color_id] Moving {} labeled images to new training_folder {}".format(len(sqlresult),
                                                                                                       dest_folder))
        self.image_counter = 0
        for row in self.progressBar(sqlresult):
            path = os.path.join('/home/robert/LegoImageCropper/', row[0])
            color_id = row[1]

            self.copy_image(path, os.path.join(dest_folder, 'color_id'), str(color_id), False)

        print("INFO: [color_id] Wrote {} image files. Skipped {}.".format(self.image_counter, self.images_skipped))

    def create_training_dir_partno(self, dest_folder, reduce_partno):
        self.cur.execute("""SELECT path, p.partno FROM LegoSorterDB.Partimages i 
        LEFT JOIN LegoSorterDB.Identifiedparts p ON p.id = i.part_id
        WHERE i.deleted IS NULL AND p.deleted IS NULL""")

        sql_result = self.cur.fetchall()
        print("INFO: [partno] Moving {} labeled images to new training_folder {}".format(len(sqlresult), dest_folder))

        for row in self.progressBar(sql_result):
            path = os.path.join('/home/robert/LegoImageCropper/', row[0])
            partno = row[1]

            self.image_counter = 0
            self.copy_image(path, os.path.join(dest_folder, 'partno'), str(partno), reduce_partno)

        print("INFO: [partno] Wrote " + str(self.image_counter) + " image files. Skipped " + str(
            self.images_skipped) + " Files")

    def copy_image(self, imagepath, dest_folder, label, reduce_partno):

        if reduce_partno:
            long_dest_folder = os.path.join(dest_folder, self.brick_basename_sep(label)[0])
        else:
            long_dest_folder = os.path.join(dest_folder, label)

        if not os.path.exists(long_dest_folder):
            # print("    Creating partdestfolder: " + long_dest_folder)
            os.makedirs(long_dest_folder)

        file_destination = os.path.join(long_dest_folder, os.path.basename(imagepath))
        if not Path(file_destination).is_file():
            copyfile(imagepath, file_destination)
            # print("    Writing file: " + file_destination)
            self.image_counter += 1
        else:
            # print("    File already exists: " + file_destination)
            self.images_skipped += 1

    @staticmethod
    def progressBar(iterable, prefix='Progress', suffix='Moved', decimals=1, length=50, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iterable    - Required  : iterable object (Iterable)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)

        # Progress Bar Printing Function
        def printProgressBar(iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

        # Initial Call
        printProgressBar(0)
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        # Print New Line on Complete
        print()
