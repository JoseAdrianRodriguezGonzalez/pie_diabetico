from sklearn.model_selection import train_test_split
def stratified(*,images:list[str],labels:list[str],test_size:float=0.10,val_size:float=0.20,seed:int=42,**_:object):
    X_train,X_test,y_train,y_test=train_test_split(images,labels
                            ,test_size=test_size,stratify=labels,random_state=seed)
    val_relative = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_relative,
        stratify=y_train, 
        random_state=seed 
    )

    return {
        "train": {"images": X_train, "labels": y_train},
        "val":   {"images": X_val,   "labels": y_val},
        "test":  {"images": X_test,  "labels": y_test},
    }

