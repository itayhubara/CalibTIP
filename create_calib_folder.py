
import os
import shutil

basepath = '/home/Datasets/imagenet/train/'
basepath_calib = '/home/Datasets/imagenet/calib/'

directory = os.fsencode(basepath)
os.mkdir(basepath_calib)
for d in os.listdir(directory):
    dir_name = os.fsdecode(d)
    dir_path = os.path.join(basepath,dir_name)
    dir_copy_path = os.path.join(basepath_calib,dir_name)
    os.mkdir(dir_copy_path)
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path,f)
        copy_file_path = os.path.join(dir_copy_path,f)
        shutil.copyfile(file_path, copy_file_path)
        break