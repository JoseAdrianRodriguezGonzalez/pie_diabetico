
import matplotlib.pyplot as plt

def curves(history: dict):
    plt.figure(figsize=(10,4))
    diccionario={k:v for k,v in history.items() if isinstance(history[k],list)}
    for name, values in diccionario.items():
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, values, label=name.replace("_", " ").title())

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

