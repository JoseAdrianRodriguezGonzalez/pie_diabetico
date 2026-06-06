import os
import pandas as pd 
def write_info(src,diccionario):
    with open(src,"w",encoding="utf-8") as f:
        f.write("--------\n")
        for elemento in diccionario:
            f.write(f"{elemento} -----> {diccionario[elemento]}\n")
        f.write("--------\n")
    print("archivo escrito con exito y guardado en ",src)
def metrics_final(src,input):
    cm = input["cm"]

    tn, fp, fn, tp = cm.ravel()
    df=pd.DataFrame({"test_acc": input["test_acc"],
        "test_f1_score": input["test_f1_score"],
        "tn": [tn],
        "fp": [fp],
        "fn": [fn],
        "tp": [tp],
        "auc_roc":input["test_auc_roc"],
        "auc_pr":input["test_auc_pr"]})
    df.to_csv(src,index=False)
