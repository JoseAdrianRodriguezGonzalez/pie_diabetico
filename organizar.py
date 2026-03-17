import os
import pandas as pd 
def extract_files(src_root):
    for folder in sorted(os.listdir(src_root)):
        path_test=os.path.join(src_root,folder)
        for model in sorted(os.listdir(path_test)):
            model_path=os.path.join(path_test,model)
            items = os.listdir(model_path)
            subdirs = [d for d in items if os.path.isdir(os.path.join(model_path, d))] 
            if subdirs:
                diccionario={"test_acc":[],
                            "test_f1_score":[],
                             "submodel":[]}
                for submodel in sorted(subdirs):
                    submodel_path = os.path.join(model_path, submodel)
                    df=pd.read_csv(os.path.join(submodel_path,"test_metrics.csv"))
                    diccionario["test_acc"].append(df["test_acc"].iloc[0])
                    diccionario["test_f1_score"].append(df["test_f1_score"].iloc[0])
                    diccionario["submodel"].append(submodel)
                df_dicc=pd.DataFrame(diccionario)
                df_dicc.to_csv(os.path.join(model_path,"test_metrics.csv"),
                               index=False)
def join_global_metrics(src_root):
    rows=[]
    for folder in sorted(os.listdir(src_root)):
        path_test=os.path.join(src_root,folder)
        for model in sorted(os.listdir(path_test)):
        
            model_path=os.path.join(path_test,model)
            df=pd.read_csv(os.path.join(model_path,"test_metrics.csv"))
            df["model"]=model
            df["fold"]=folder 
            rows.append(df)
    return pd.concat(rows,ignore_index=True )
def separate_models(df:pd.DataFrame,src_root):
    models=df["model"].unique()
    for model in models:
        subdf=df[df["model"]==model]
        subdf.to_csv(os.path.join(src_root,model+".csv"))
def separate_excel(csv_path,output_dir):
    df=pd.read_csv(csv_path)
    os.makedirs(output_dir,exist_ok=True)
    for model, df_models in df.groupby("model"):
        file_path=os.path.join(output_dir,f"{model}.xlsx")
        with pd.ExcelWriter(file_path,engine="openpyxl") as writer:
            df["submodel"]=df["submodel"].astype(str)
            for submodel,df_sub in df.groupby("submodel"):
                df_sub=df_sub.sort_values("fold")
                print(submodel)
                df_sub.to_excel(writer,sheet_name=submodel,index=False)
def analyze_excels(excel_dir):
    results = []
    for file in os.listdir(excel_dir):
        if not file.endswith(".xlsx"):
            continue
        model_name = file.replace(".xlsx", "")
        file_path = os.path.join(excel_dir, file)
        sheets = pd.read_excel(file_path, sheet_name=None)
        for submodel, df in sheets.items():
            if "test_acc" not in df.columns:
                continue
            stats = {
                "model": model_name,
                "submodel": submodel,
                "mean_acc": df["test_acc"].mean(),
                "std_acc": df["test_acc"].std(),
                "min_acc": df["test_acc"].min(),
                "max_acc": df["test_acc"].max(),
                "mean_f1": df["test_f1_score"].mean(),
                "std_f1": df["test_f1_score"].std(),
                "n_folds": len(df)
            }
            results.append(stats)
    return pd.DataFrame(results)
extract_files("out/tests/")
df=join_global_metrics("out/tests/")
separate_models(df,"out/global_metric")
for model in os.listdir("out/global_metric"):
    path=os.path.join("out/global_metric/",model)
    if not os.path.isdir(path):
        separate_excel(path,"out/global_metric/excel/")
df=analyze_excels("out/global_metric/excel/")
df["robust_acc"]=df["mean_acc"]-df["std_acc"]
df["robust_f1"]=df["mean_f1"]-df["std_f1"]
df.to_csv("make_unique.csv")

