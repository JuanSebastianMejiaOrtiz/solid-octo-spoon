import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import scipy.fftpack as fftpack


'''
t0: tiempo inicial
t1: tiempo final
fs: frecuencia de muestreo
audio_og: audio original
'''
def get_audio_fragment(audio_og, fs, t0, t1):
    start_sample = int(t0 * fs) # muestra inicial
    end_sample = int(t1 * fs) # muestra final
    audio_fragment = audio_og[start_sample:end_sample].copy() # el audio puede ser tratado como una lista
    t_frag = np.linspace(t0, t1, end_sample - start_sample)
    return t_frag, audio_fragment


def windowing(audio_fragment, fs, tv):
    N = int(fs * tv)
    num_windows = len(audio_fragment) // N

    windows = []
    for i in range(num_windows):
        start = i * N
        end = start + N
        window = audio_fragment[start:end]
        window = window * np.hanning(N)
        windows.append(window)

    return windows


def plot_audio(t_frag, audio_fragment):
    plt.figure()
    plt.plot(t_frag, audio_fragment)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Audio')


def distance(x, y):
    return np.linalg.norm(x - y)


def cepstogram(dct):
    plt.figure()
    plt.imshow(dct.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Cepstogram')
    plt.xlabel('Windows')
    plt.ylabel('MFCCs')


def process_audio(audio_file):
    fs, data = wav.read(audio_file)
    # print(data.shape)

# Parameters
    duration = len(data)/fs
    tv = 25e-3
    hop_ms = 10e-3
    NFFT = int(2 * fs * tv)
    sample_hop = int(fs * hop_ms)

    # print(f'f_s = {fs}')
    # print(f'd = {duration}')

# PREPROCESSING
        # Normalization
    mag = np.max(np.abs(data))
    data = data/mag

        # Truncation
    t_frag, audio_fragment = get_audio_fragment(data, fs, 0.6, 2.9)

# WINDOWING
    windows = windowing(audio_fragment, fs, tv)

# DFT
    fft = np.fft.fft(windows)
    fft = 10 * np.log10(np.abs(fft)**2 + 1e-15)

# MEL BANKS
    n_mels = 24
    mel_fb = librosa.filters.mel(sr=fs, n_fft=NFFT, n_mels=n_mels)[:, :fft.shape[1]]

    mel = np.matmul(fft, mel_fb.T)

# DCT
    dct = fftpack.dct(mel, type=2, axis=1, norm='ortho')
    dct = dct[:, :mel.shape[1]//2 + 1]

    # print(dct.shape)

# CHARACTERISTICS
    char = np.mean(dct, axis=0)

    # print(f'char.shape = {char.shape}')
    # print(char)
    return char
