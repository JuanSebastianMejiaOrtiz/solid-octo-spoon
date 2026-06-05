from pydub import AudioSegment
import Codigo.functions.general as gen

a_mean = gen.firstgroup('a')
i_mean = gen.firstgroup('i')
u_mean = gen.firstgroup('u')

is_ogg = True
# audio = input('Audio: ')
# is_ogg = Bool(input('Ogg: '))
audio = 'predict'
audio_wav = ''

if is_ogg:
    audio_ogg = f'audio_predict/{audio}.ogg'
    audio_wav = audio_ogg.replace('.ogg', '.wav')
    audio = AudioSegment.from_ogg(audio_ogg)
    audio.export(audio_wav, format="wav")
else:
    audio_wav  = f'audio_predict/{audio}.wav'

predict = gen.classify(audio_wav, a_mean, i_mean, u_mean)

print('Predict: ', predict)
