from sklearn.model_selection import train_test_split
def stratified(*,images:list[str],labels:list[str],test_size:float=0.2,seed:int=42,**_:object):
    return train_test_split(images,labels
                            ,test_size=test_size,stratify=labels,random_state=seed)

