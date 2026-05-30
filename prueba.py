import Codigo.functions.general as gen


a_char = []
i_char = []
u_char = []
for i in range(1, 17 + 1):
    audio_file = f'Audios/Hombre/a/a_{i}.wav'
    a_char.append(gen.process_audio(audio_file))

for i in range(1, 17 + 1):
    audio_file = f'Audios/Mujer/a/a_{i}.wav'
    a_char.append(gen.process_audio(audio_file))

for i in range(1, 17 + 1):
    audio_file = f'Audios/Hombre/i/i_{i}.wav'
    i_char.append(gen.process_audio(audio_file))

for i in range(1, 17 + 1):
    audio_file = f'Audios/Mujer/i/i_{i}.wav'
    i_char.append(gen.process_audio(audio_file))

for i in range(1, 17 + 1):
    audio_file = f'Audios/Hombre/u/u_{i}.wav'
    u_char.append(gen.process_audio(audio_file))

for i in range(1, 17 + 1):
    audio_file = f'Audios/Mujer/u/u_{i}.wav'
    u_char.append(gen.process_audio(audio_file))


# Salida
for i in range(1, 17 * 2):
    distance = gen.distance(a_char[0], a_char[i])
    print(f'a_0 con a_{i}: {distance}')
print('-' * 50)

for i in range(1, 17 * 2):
    distance = gen.distance(a_char[0], i_char[i])
    print(f'a_0 con i_{i}: {distance}')
print('-' * 50)

for i in range(1, 17 * 2):
    distance = gen.distance(a_char[0], u_char[i])
    print(f'a_0 con u_{i}: {distance}')
print('-' * 50)
