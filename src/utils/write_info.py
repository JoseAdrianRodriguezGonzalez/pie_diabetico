import os
def write_info(src,diccionario):
    with open(src,"w",encoding="utf-8") as f:
        f.write("--------\n")
        for elemento in diccionario:
            f.write(f"{elemento} -----> {diccionario[elemento]}\n")
        f.write("--------\n")
    print("archivo escrito con exito y guardado en ",src)
