#!/usr/bin/python
import pymysql
import os
from shutil import copyfile
from pathlib import Path
from tqdm import tqdm


class ImageMover():
    def __init__(self):
        self.image_counter = 0
        self.images_skipped = 0
        self.db = pymysql.connect(host="mysqlserver",  # your host, usually localhost
                                  user="WebDBUser",  # your username
                                  passwd="qF2J%9a84zU",  # your password
                                  db="LegoSorterDB")  # name of the data base
        self.cur = self.db.cursor()

    def move_images(self, dest_folder):
        self.cur.execute("""SELECT path, camera, p.partno, p.color_id, camera FROM LegoSorterDB.Partimages i 
        LEFT JOIN LegoSorterDB.Identifiedparts p ON p.id = i.part_id
        WHERE i.deleted IS NULL AND p.deleted IS NULL""")

        for row in self.cur.fetchall():
            path = os.path.join('/home/robert/LegoImageCropper/', row[0])
            camera = row[1]
            partno = row[2]
            color_id = row[3]

            self.copy_image(path, os.path.join(dest_folder, 'partno'), str(partno))
            self.copy_image(path, os.path.join(dest_folder, 'color_id', camera), str(color_id))

        print("Wrote " + str(self.image_counter) + " image files. Skipped " + str(self.images_skipped) + " Files")

    def copy_image(self, imagepath, dest_folder, label):

        long_dest_folder = os.path.join(dest_folder, label)
        if not os.path.exists(long_dest_folder):
            print("    Creating partdestfolder: " + long_dest_folder)
            os.makedirs(long_dest_folder)

        file_destination = os.path.join(long_dest_folder, os.path.basename(imagepath))
        if not Path(file_destination).is_file():
            copyfile(imagepath, file_destination)
            print("    Writing file: " + file_destination)
            self.image_counter += 1
        else:
            print("    File already exists: " + file_destination)
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
