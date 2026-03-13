import os 
def make_dir_test(src,models,submodels):
    os.makedirs(src,exist_ok=True)
    for m in models:
        path_model=os.path.join(src,m)
        os.makedirs(path_model,exist_ok=True)
        if len(submodels)!=0:
            for s in submodels[m]:
                path_submodels=os.path.join(path_model,s)
                os.makedirs(path_submodels,exist_ok=True)
        
