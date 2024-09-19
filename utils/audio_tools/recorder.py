import os
import random
import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

from count_files import count_files_in_folders

def grabar_audio(nombre_archivo, duracion=1, frecuencia_muestreo=16000):
    print("Grabando...")
    # Graba el audio con la duración y frecuencia de muestreo especificadas
    audio = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='int16')
    
    # Espera hasta que termine la grabación
    sd.wait()
    
    # Guarda el archivo WAV
    write(nombre_archivo, frecuencia_muestreo, audio)
    print(f"Grabación guardada como {nombre_archivo}")

def generar_numero_unico(numeros_usados, max_num=1000):
    while True:
        numero = random.randint(1, max_num)
        if numero not in numeros_usados:
            numeros_usados.add(numero)
            return numero

# Crear un conjunto para almacenar números únicos usados
numeros_usados = set()

# Generar un número único
numero_unico = generar_numero_unico(numeros_usados)

# Crear el nombre del archivo
fileName = f"data/dataset/hola/{numero_unico}.wav"
# fileName = f"data/testing/ruido/{numero_unico}.wav"
# fileName = f"{numero_unico}.wav"

grabar_audio(fileName)
count_files_in_folders('data/dataset')

# while True:
#     folder_path = 'data/dataset/ruido'
    
#     if os.path.isdir(folder_path):
#         file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
#         print(f"Carpeta: ruido, Cantidad de archivos: {file_count}")
        
#         if file_count == 150:
#             break
    
#     # Genera un número único para el nombre del archivo
#     numero_unico = random.randint(100000, 999999)
#     fileName = f"data/dataset/ruido/{numero_unico}.wav"
#     grabar_audio(fileName)
    
#     # Añadir un pequeño retraso
#     time.sleep(1)