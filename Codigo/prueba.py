from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

i = 1
audio_file = f'../Audios/Hombre/a/a_{i}.wav'
fs, data = read(audio_file)
duration = len(data)/fs

# Normalization
mag = np.max(np.abs(data))
data = data/mag

print(f'f_s = {fs}')
print(f'd = {duration}')

plt.figure()
plt.title('Escala lineal')
plt.specgram(data, Fs=fs, scale='linear')

plt.figure()
plt.title('Escala Logaritmica')
plt.specgram(data, Fs=fs, scale='dB')

plt.show()
