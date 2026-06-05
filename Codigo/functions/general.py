import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import scipy.spatial as spp
import pandas as pd
import seaborn as sns


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


def firstgroup(letra):
    vowel = []
    for genero in ['Hombre', 'Mujer']:
        for i in range(1, 10 + 1):
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            vowel.append(process_audio(audio_file))
    vowel = np.asarray(vowel)
    vowel = np.mean(vowel, axis=0)
    return vowel


def cosine_distance(vowel_mean, letra):
    distance = []
    for genero in ['Hombre', 'Mujer']:
        for i in range(11, 15 + 1):        
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            vowel_rep=process_audio(audio_file)
            distance.append(spp.distance.cosine(vowel_rep,vowel_mean))
    return distance


def stat_test(vowel1, vowel2, alpha=0.05):
    test = shapiro(vowel1)
    p1 = test.pvalue
    test = shapiro(vowel2)
    p2 = test.pvalue
    if (p1 > alpha and p2 > alpha):
        print("both follow a normal distribution")
        test = ttest_ind(vowel1, vowel2)
        pval = test.pvalue
        if(pval > alpha):
            print("Both mean are the same")
        else:
            print("Means are different")
    else:
        print("At least one does not follow a normal distribution")
        test = mannwhitneyu(vowel1, vowel2)
        pval = test.pvalue
        if (pval > alpha):
            print("Distributions are similars")
        else:
            print("Distributions are different")


def cepstogram(dct):
    plt.figure()
    plt.imshow(dct.T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Cepstogram')
    plt.xlabel('Windows')
    plt.ylabel('MFCCs')


def process_audio(audio_file):
    fs, data = wav.read(audio_file)

# Parameters
    duration = len(data)/fs
    tv = 25e-3
    hop_ms = 10e-3
    NFFT = int(2 * fs * tv)
    sample_hop = int(fs * hop_ms)

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

        # CHARACTERISTICS
    char = np.concatenate([np.mean(dct, axis=0), np.std(dct, axis=0)])
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 27, 35, 36, 47]
    char = np.concatenate([ char[0:16], char[17:21], char[22:24], [char[27]], char[35:37], [char[47]] ])
    return char


def classify(audio_file, a_mean, i_mean, u_mean):
    processed_audio = process_audio(audio_file)
    dist_a = spp.distance.cosine(processed_audio, a_mean)
    dist_i = spp.distance.cosine(processed_audio, i_mean)
    dist_u = spp.distance.cosine(processed_audio, u_mean)
    if dist_a < dist_i and dist_a < dist_u:
        return 'A'
    elif dist_i < dist_a and dist_i < dist_u:
        return 'I'
    else:
        return 'U'


'''
    Segun como esta cargado deberia de salir:
    + Primer ciclo:
        - Hombre 16: A
        - Hombre 17: A
        - Hombre 18: A
        - Hombre 19: A
        - Mujer 16:  A
        - Mujer 17:  A
    + Segundo ciclo:
        - Hombre 16: I
        - Hombre 17: I
        - Hombre 18: I
        - Hombre 19: I
        - Mujer 16:  I
        - Mujer 17:  I
    + Tercer ciclo:
        - Hombre 16: U
        - Hombre 17: U
        - Hombre 18: U
        - Hombre 19: U
        - Mujer 16:  U
        - Mujer 17:  U
'''
def lastgroup(letra, a_mean, i_mean, u_mean):
    predicts = []
    for i in range(16, 19 + 1):
        audio_file = f'Audios/Hombre/{letra}/{letra}_{i}.wav'
        predict = classify(audio_file, a_mean, i_mean, u_mean)
        predicts.append(predict)

    for i in range(16, 17 + 1):
        audio_file = f'Audios/Mujer/{letra}/{letra}_{i}.wav'
        predict = classify(audio_file, a_mean, i_mean, u_mean)
        predicts.append(predict)

    return predicts
