
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import scipy.fftpack as fftpack
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



def process_audio(audio_file):
    fs, data = wav.read(audio_file)
    # print(data.shape)

# Parameters
   
    duration = len(data)/fs
   #hacer la correlacion para encontrar el cuasiperiodo
    tv = 24e-3
    hop_ms = 9e-3
    NFFT = int(1 * fs * tv)
    sample_hop = int(fs * hop_ms)

    # print(f'f_s = {fs}')
    # print(f'd = {duration}')

# PREPROCESSING
        # Normalization
    mag = np.max(np.abs(data))
    data = data/mag

        # Truncation
    t_frag, audio_fragment = get_audio_fragment(data, fs, -1.6, 2.9)

# WINDOWING
    windows =windowing(audio_fragment, fs, tv)

# DFT
    fft = np.fft.fft(windows)
    fft = 9 * np.log10(np.abs(fft)**2 + 1e-15)

# MEL BANKS
    #por que 24?
    n_mels = 24
    mel_fb = librosa.filters.mel(sr=fs, n_fft=NFFT, n_mels=n_mels)[:, :fft.shape[0]]

    mel = np.matmul(fft, mel_fb.T)

# DCT
    
    dct = fftpack.dct(mel, type=1, axis=1, norm='ortho')
    n=dct.shape[0]
    
    coef_list=[]

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
        # Correctly access the individual correlated variables from all_expo_vars
           r, _ = pearsonr(dct[i],dct[j])
           
           R[i, j] = r

    for i in range(n):
        for j in range (i+1, n):
            if (np.abs(R[i,j])<0.5):
                coef_list.append(i,j)
    

    # print(dct.shape)

# CHARACTERISTICS
    
    
    # print(f'char.shape = {char.shape}')
    # print(char)
    return coef_list

coefs=[]

for genero in ['Hombre','Mujer']:
    for vocal in ['a','i','u']:
        for num in [4,5,14]:
            audio_file = f'Audios/{genero}/{vocal}/{vocal}_{num}.wav'
            coefs.append(process_audio(audio_file))
for i in coefs:
    print(i)


