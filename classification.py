from pydub import AudioSegment
import os
import Codigo.functions.general as gen

a_mean = gen.firstgroup('a')
i_mean = gen.firstgroup('i')
u_mean = gen.firstgroup('u')

# audio_file = input('Audio: ')
audio = 'predict17.mp4'
audio_file = f'audio_predict/{audio}'
base, ext = os.path.splitext(audio_file)
audio_wav = base + '.wav'

if ext.lower() != '.wav':
    audio = AudioSegment.from_file(audio_file)  # soporta mp4, ogg, etc.
    audio.export(audio_wav, format="wav")
    print(f"Conversión completa: {audio_wav}")

predict = gen.classify(audio_wav, a_mean, i_mean, u_mean)

print('Predict: ', predict)
