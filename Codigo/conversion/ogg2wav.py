from pydub import AudioSegment

for genero in ["Hombre", "Mujer"]:
    for letra in ["a", "i", "u"]:
        for i in range(1, 18 + 1):
            if i == 18 and genero != "Hombre":
                continue
            audio = AudioSegment.from_ogg(f"Audios/{genero}/{letra}_ogg/{letra}_{i}.ogg")
            audio.export(f"Audios/{genero}/{letra}/{letra}_{i}.wav", format="wav")
