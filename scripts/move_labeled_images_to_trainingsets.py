#usage:  python mergeImagesWithRenderedData.py -ren 0 -fotos 1 -folder /home/robert/LegoSorter/training_images/dataset_20200723_onlyFotos/
#!/usr/bin/python
import pymysql
import os
import os.path
import argparse
from shutil import copyfile
import pathlib
import argparse
from pathlib import Path

db = pymysql.connect(host="localhost",    # your host, usually localhost
                    user="WebDBUser",         # your username
                    passwd="qF2J%9a84zU",  # your password
                    db="LegoSorterDB")        # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()
imagecounter = 0
imageskipped = 0
imagebasepath = "/home/robert/LegoCollectionManager/frontend/src"
trainingsetname = "trainingset_1"

def getImgesIntoSortedPartimages_Folder():
    global imagecounter, imageskipped, imagebasepath
    print("Writing images into Sorted_Partimages folder.")
    # Use all the SQL you like
    cur.execute("""SELECT ip.no, ip.color_id, pi.path
                FROM LegoSorterDB.Identifiedparts ip
                LEFT JOIN LegoSorterDB.Identifiedimages ii ON ii.part_id = ip.id 
                LEFT JOIN LegoSorterDB.Partimages pi ON pi.id = ii.image_id
                WHERE ip.identifier = 'human' and ip.deleted IS NULL
                AND pi.deleted IS NULL""")

    destinationfolderpartno = "./../trainingsets/partno"
    destinationfoldercolorid = "./../trainingsets/colorid"
    for row in cur.fetchall():
        imagepath = os.path.join(imagebasepath,row[2])
        print("    Copy image:" + imagepath)
        moveImageToSortedPartimages( imagepath,destinationfolderpartno,str(row[0]))        
        moveImageToSortedPartimages( imagepath,destinationfoldercolorid,str(row[1]))        
    db.close()
    print("Wrote "  + str(imagecounter) + " image files. Skipped " + str(imageskipped) + " Files")

def moveImageToSortedPartimages(imagepath, destinationfolder, foldername):
    global imagecounter, imageskipped, imagebasepath, trainingsetname
    #create sourcefolder if not exisiting
    destfolder = os.path.join(destinationfolder,trainingsetname, foldername) 
    if not os.path.exists(destfolder):
        print("    Creating folder: " + destfolder)
        os.makedirs(destfolder)
   
    fileto = os.path.join(destfolder ,os.path.basename(imagepath))
    print("    Writing file: " + fileto)    
    if not Path(fileto).is_file():
        copyfile(imagepath, fileto)
        print("    Writing file: " + fileto)    
        imagecounter += 1
    else:
        print("    File already exists: " + fileto)
        imageskipped += 1

def moveImages(sourcefolder, finalDestFolder):
    
    for dirpath, _, filenames in os.walk(sourcefolder):
        for filename in [f for f in filenames if f.endswith(".png") or f.endswith(".jpg")]:
            filefrom = os.path.join(dirpath, filename)

            partdestfolder = os.path.join(finalDestFolder,os.path.basename(dirpath) ) 
            if not os.path.exists(partdestfolder):
                print("    Creating partdestfolder: " + partdestfolder)
                os.makedirs(partdestfolder)

            fileto = os.path.join(partdestfolder,filename) 
            print("    Writing file from " + filefrom + " to " + fileto)    
            copyfile(filefrom, fileto)

def main():
    getImgesIntoSortedPartimages_Folder()

main()