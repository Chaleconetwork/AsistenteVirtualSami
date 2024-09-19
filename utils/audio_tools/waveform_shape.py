import librosa

# Cargar el archivo de audio
# audio_path = 'data/mini_speech_commands/right/0ea0e2f4_nohash_0.wav'
# audio_path = 'dataset/hola/902.wav'
audio_path = 'p_28839811_663.wav'
waveform, sr = librosa.load(audio_path, sr=None)  # sr=None para mantener la frecuencia de muestreo original

# Imprimir el shape del waveform
print(waveform.shape)
