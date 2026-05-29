from pydub import AudioSegment

for genero in ["Hombre", "Mujer"]:
    for letra in ["a", "i", "u"]:
        for i in range(1, 17 + 1):
            audio = AudioSegment.from_ogg(f"{genero}/{letra}_ogg/{letra}_{i}.ogg")
            audio.export(f"{genero}/{letra}/{letra}_{i}.wav", format="wav")
