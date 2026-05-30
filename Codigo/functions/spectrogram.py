import numpy as np
import matplotlib.pyplot as plt
import librosa


def make_spectrograms_all_scales(segm, fs, NFFT, sample_hop, export=False, scales = ['linear', 'log', 'mel'],pathFile = 'spectrogram.pdf', ylimmin=0, ylimmax=None, xlimmin=0, xlimmax=None):

    plt.figure(figsize=(5*len(scales),4))
    for i,scale in enumerate(scales):
        stft = np.abs(librosa.stft(segm,  n_fft=NFFT, hop_length=sample_hop))
        if scale == 'mel':
            melfb = librosa.filters.mel(sr=fs, n_fft=NFFT, n_mels=128, fmax=8000)
            stft_mel = np.matmul(melfb, stft)
            plt.subplot(1,len(scales),i+1)
            D = librosa.amplitude_to_db(stft_mel, ref = np.max) # Se convierte la amplitud de lineal a dB
            spec = librosa.display.specshow(D, sr=fs, x_axis='time', y_axis=scale, cmap=None, hop_length=sample_hop)
        else:
            plt.subplot(1,len(scales),i+1)
            D = librosa.amplitude_to_db(stft, ref = np.max) # Se convierte la amplitud de lineal a dB
            spec = librosa.display.specshow(D, sr=fs, x_axis='time', y_axis=scale, cmap=None, hop_length=sample_hop)

        plt.title('Espectrograma ' + scale)
        if ylimmax != None:
            plt.ylim(ylimmin, ylimmax)
        if xlimmax != None:
            plt.xlim(xlimmin, xlimmax)


    if(export):
        plt.savefig(pathFile)
