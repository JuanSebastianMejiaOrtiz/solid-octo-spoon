import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import scipy.fftpack as fftpack
from scipy.stats import shapiro, f_oneway, kruskal
import Codigo.functions.general as gen


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


def stat_test():
    pass


def process_audio_full(audio_file):
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
    # fft = 10 * np.log10(np.abs(fft)**2 + 1e-15)
    power = np.abs(fft)**2

# MEL BANKS
    n_mels = 24
    mel_fb = librosa.filters.mel(sr=fs, n_fft=NFFT, n_mels=n_mels)[:, :fft.shape[1]]

    mel = power @ mel_fb.T
    mel = 10 * np.log10(mel + 1e-15)

# DCT
    dct = fftpack.dct(mel, type=2, axis=1, norm='ortho')

# CHARACTERISTICS
    char = np.concatenate([np.mean(dct, axis=0), np.std(dct, axis=0)])
    return char


def firstgroup(letra):
    vowel = []
    for genero in ['Hombre', 'Mujer']:
        for i in range(1, 10 + 1):
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            vowel.append(process_audio_full(audio_file))
    vowel = np.asarray(vowel)
    return vowel
    

def test_coeff(vowel1, vowel2, vowel3, alpha=0.05):
    test = shapiro(vowel1)
    p1 = test.pvalue
    test = shapiro(vowel2)
    p2 = test.pvalue
    test = shapiro(vowel3)
    p3 = test.pvalue

    pval = 0
    if (p1 > alpha and p2 > alpha and p3 > alpha):
        print("All follow a normal distribution")
        test = f_oneway(vowel1, vowel2, vowel3) # ANOVA
        pval = test.pvalue
        if(pval > alpha):
            print("All mean are the same")
        else:
            print("Means are different")
    else:
        print("At least one does not follow a normal distribution")
        test = kruskal(vowel1, vowel2, vowel3) # Kruskal
        pval = test.pvalue
        if (pval > alpha):
            print("Distributions are similars")
        else:
            print("Distributions are different")
    return pval < alpha
    

chars = []
for vocal in ['a','i','u']:
    chars.append(firstgroup(vocal))
chars = np.asarray(chars) # 3x20x24

print(f'chars.shape: {chars.shape}')

charsa = chars[0].T # 24x20
charsi = chars[1].T # 24x20
charsu = chars[2].T # 24x20

coeffs_pos = []
for i in range(charsa.shape[0]):
    vowel1 = charsa[i]
    vowel2 = charsi[i]
    vowel3 = charsu[i]
    if (test_coeff(vowel1, vowel2, vowel3)):
        coeffs_pos.append(i)

print(coeffs_pos)
