import seaborn as sns 
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
def curves(tupla_values,nombres,model_name="Model",path="out"):
    plt.figure(figsize=(10,4))
    for valores,nombres in zip(tupla_values,nombres):
        epochs = list(range(1, len(valores) + 1))
        plt.plot(epochs, valores, label=nombres.replace("_", " ").title())
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    plot_path = os.path.join(path,f"{model_name}_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.close() 
def plot_confussion_matrix(cm,classes,model_name="Model",path="out"): 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.xlabel("Predicciones")
    plt.ylabel("Reales")
    plt.tight_layout()
    plot_path = os.path.join(path, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[INFO] Matriz de confusión guardada en {plot_path}")
