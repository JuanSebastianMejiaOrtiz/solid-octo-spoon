import Codigo.functions.spectrogram as sg
import Codigo.functions.general as gen
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


def specgrams(letra):
    for genero in ['Hombre', 'Mujer']:
        for i in range(1, 3 + 1):
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            fs, data = wav.read(audio_file)
            duration = len(data)/fs
            tv = 25e-3
            NFFT = int(2 * fs * tv)
            hop_ms = 10e-3
            sample_hop = int(fs * hop_ms)

                # Normalization
            mag = np.max(np.abs(data))
            data = data/mag

                # Truncation
            _, audio_fragment = gen.get_audio_fragment(data, fs, 0.6, 2.9)

            sg.make_spectrograms_all_scales(audio_fragment, fs, NFFT, sample_hop)
            plt.suptitle(f'{genero} letra:{letra}; numero:{i}')


for letra in ['a', 'i', 'u']:
    specgrams(letra)

plt.show()
