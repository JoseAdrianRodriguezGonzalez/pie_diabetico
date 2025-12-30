import os
from collections import Counter
from os.path import isdir
def images(src,exts=("jpg","png","jpeg")):
    """Given a certain datasets, it will divide the and count the quantity of samples
    """
    db={}
    for folder in os.listdir(src):
        new_path=os.path.join(src,folder)
        if not os.path.isdir(new_path):
            continue 
        files= [image for image in os.listdir(new_path) if image.lower().endswith(exts)]
        db[folder]=len(files)
    return db

