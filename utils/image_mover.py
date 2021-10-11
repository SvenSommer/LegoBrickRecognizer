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

    def is_flip(self, file_path: str) -> bool:
        view_type = file_path.split('_')[-1].split('.')[0]
        return view_type in ['LEFT', 'RIGHT']

    def move_images(self, dest_folder):
        self.cur.execute("""SELECT path, camera, p.partno, p.color_id, c.color_type FROM LegoSorterDB.Partimages i 
        LEFT JOIN LegoSorterDB.Identifiedparts p ON p.id = i.part_id
        LEFT JOIN LegoSorterDB.Colors c ON p.color_id = c.color_id
        WHERE i.deleted IS NULL AND p.deleted IS NULL""")

        sqlresult = self.cur.fetchall()
        print("INFO: Moving {} labeled images to new training_folder {}".format(len(sqlresult), dest_folder))

        for row in self.progressBar(sqlresult):
            path = os.path.join('/home/robert/LegoImageCropper/', row[0])
            camera = row[1]
            partno = row[2]
            color_id = row[3]
            color_type = row[4]

            self.copy_image(path, os.path.join(dest_folder, 'partno'), str(partno))
            self.copy_image(path, os.path.join(dest_folder, 'color_id', camera), str(color_id))
            self.copy_image(path, os.path.join(dest_folder, 'color_type', camera, str(color_type)), str(color_id))

        print("INFO: Wrote " + str(self.image_counter) + " image files. Skipped " + str(self.images_skipped) + " Files")

    def copy_image(self, imagepath, dest_folder, label):

        long_dest_folder = os.path.join(dest_folder, self.brick_basename_sep(label)[0])
        if not os.path.exists(long_dest_folder):
            # print("    Creating partdestfolder: " + long_dest_folder)
            os.makedirs(long_dest_folder)

        file_destination = os.path.join(long_dest_folder, os.path.basename(imagepath))
        if not Path(file_destination).is_file():
            if self.is_flip(imagepath):
                image = Image.open(imagepath).convert('RGB')
                image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
                image.save(file_destination)
                # print("    Flipped image: " + file_destination)
            else:
                copyfile(imagepath, file_destination)
            # print("    Writing file: " + file_destination)
            self.image_counter += 1
        else:
            # print("    File already exists: " + file_destination)
            self.images_skipped += 1

    def split_train_test_dataset(self, base_folder):
        train_folder = os.path.join(base_folder, 'partno/')
        test_folder = os.path.join(base_folder, 'partno_val/')

        classes_count = 0
        for folder in tqdm(os.listdir(train_folder)):
            res_folder = os.path.join(test_folder, folder)
            os.makedirs(res_folder, exist_ok=True)
            k = 0
            classes_count += 1
            for file_name in sorted(os.listdir(os.path.join(train_folder, folder))):
                if k > 5:
                    continue
                k += 1
                target_file = os.path.join(train_folder, folder, file_name)
                save_file = os.path.join(res_folder, file_name)
                copyfile(target_file, save_file)
                os.remove(target_file)
        return classes_count

    def brick_basename_sep(self, bname: str) -> list:
        """
        Separate brick name to base and additional names
        Args:
            bname: string
        Returns:
            List with names
        Examples:
            brick_basename_sep('283bc1') -> ['283', 'bc1']
            brick_basename_sep('283211') -> ['283211']
            brick_basename_sep('u238') -> ['u238']
        """
        name_char_i = 0
        while True:
            if name_char_i >= len(bname):
                break
            if not bname[name_char_i].isnumeric():
                if name_char_i == 0:
                    name_char_i += 1
                    continue
                break
            name_char_i += 1
        if name_char_i == len(bname):
            return [bname]
        return [bname[:name_char_i], bname[name_char_i:]]

    def progressBar(self, iterable, prefix='Progress', suffix='Moved', decimals=1, length=50, fill='█', printEnd="\r"):
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
