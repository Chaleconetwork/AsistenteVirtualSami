import os

def count_files_in_folders(data_dir):
    # Obtener los nombres de las carpetas en el directorio de datos
    folder_names = os.listdir(data_dir)
    
    # Contar archivos en cada carpeta
    for folder in folder_names:
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            print(f"Carpeta: {folder}, Cantidad de archivos: {file_count}")

data_dir = "data/dataset"
count_files_in_folders(data_dir)