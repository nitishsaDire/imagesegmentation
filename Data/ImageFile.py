import re
import os

class ImageFiles:
    '''It gives all the video files names recursively in the root path along with their labels'''
    def __init__(self, image_root_path, extensions):
        self.image_root_path = image_root_path
        self.extensions = extensions


    def get_files_paths(self):
        imageFiles = []
        for path, subdirs, files in os.walk(self.image_root_path):
            for name in files:
                print(path+name)
                for e in self.extensions:
                    if name.endswith(e):
                        imageFilePath = path + "/" + name
                        imageFiles.append(imageFilePath)
        return  imageFiles