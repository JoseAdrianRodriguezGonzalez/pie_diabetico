from ingest.discover import images
from utils.write_info import  write_info
diccionario=images("../data/raw/datasets_diabetes/DFUNET/PartA_DFU_Dataset/")
diccionario["Titulo"]="Distribución de base de datos"
write_info("../out/distribution_database/PartA_DFU_Dataset.txt",diccionario)

