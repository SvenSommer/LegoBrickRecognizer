#!/usr/bin/python
import pymysql
import os
from shutil import copyfile
from pathlib import Path

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
