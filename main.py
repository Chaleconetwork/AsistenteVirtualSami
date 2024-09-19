import tensorflow as tf
import numpy as np
import time

from commands.commands import open_trigger
from utils.audio_tools.streaming_record import record_audio
from utils.tensorflow.tf_preprocess import preprocess_audiobuffer

# Cargar el modelo guardado
model = tf.keras.models.load_model('sami.keras')

# Comandos por agregar: google, stop, whatsapp, youtubemusic, 
commands = ['abre', 'hola', 'ruido', 'sami', 'youtube']

def is_audio_signal_present(audio, threshold=0.01):
    return np.max(np.abs(audio)) > threshold

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = model(spec)
    confidence = np.max(prediction)
    if confidence < 0.5:
        return None
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    return command

def listen_for_command():
    while True:
        command = predict_mic()
        if command == 'sami':
            print("Detected 'sami'. Listening for the next command...")
            return
        print("Listening for 'sami'...", end='\r')

def process_command():
    detected_commands = []
    start_time = time.time()  # Inicia el temporizador
    timeout_duration = 1  # Duración del tiempo de espera en segundos

    while True:
        # Captura el tiempo actual para detenerse después de 3 segundos
        current_time = time.time()
        
        if current_time - start_time >= timeout_duration:
            # Si ha pasado el tiempo de espera, imprime los comandos detectados y detiene la ejecución
            if detected_commands:
                print("Commands detected:", " ".join(detected_commands))
                open_trigger(detected_commands)
            else:
                print("No commands detected.")
            break  # Sale del bucle y detiene la ejecución

        command = predict_mic()
        if command in commands:
            detected_commands.append(command)
            # print(f"Detected command: {command}")

if __name__ == "__main__":
    while True:
        listen_for_command()
        process_command()