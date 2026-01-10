import os
def images(src,exts=("jpg","png","jpeg")):
    """Given a certain datasets, it will divide the and count the quantity of samples
    """
    db={}
    for folder in os.listdir(src):
        new_path=os.path.join(src,folder)
        if not os.path.isdir(new_path):
            continue 
        db[folder]=[
            os.path.join(new_path,f)
            for f in os.listdir(new_path)
            if f.lower().endswith((".jpg",".png"))
        ]
    return db
