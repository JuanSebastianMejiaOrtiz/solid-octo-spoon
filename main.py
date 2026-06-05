import Codigo.functions.general as gen
import numpy as np
import scipy.spatial as spp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from sklearn.metrics import confusion_matrix, accuracy_score


# Comprobacion edades apareadas
edadH = [21, 21, 19, 21, 25, 52, 45, 22, 25, 25, 40, 25, 40, 26, 23, 25, 50, 30, 83]
print("Hombres")
print(len(edadH))
print(sum(edadH)/len(edadH))
print('-' * 30)

edadM = [54, 78, 18, 17, 13, 14, 45, 52, 20, 20, 24, 52, 24, 45, 25, 20, 50]
print("Mujeres")
print(len(edadM))
print(sum(edadM)/len(edadM))
print('-' * 30)

df_H = pd.DataFrame({'edad': edadH, 'sexo': 'hombre'})
df_M = pd.DataFrame({'edad': edadM, 'sexo': 'mujer'})

df_combined = pd.concat([df_H, df_M])

plt.figure(figsize = (10,6))
sns.boxplot(data = df_combined, x="edad", y='sexo')
plt.title('Comparación Edades Personas')
plt.xlabel('Edad')
plt.ylabel('Sexo')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

gen.stat_test(edadH, edadM)
print('-' * 30)
print('\n')

# Grupos de entrenamiento
a_mean = gen.firstgroup('a')
i_mean = gen.firstgroup('i')
u_mean = gen.firstgroup('u')

# Entre mismas vocales
a_distance = gen.cosine_distance(a_mean, 'a')
i_distance = gen.cosine_distance(i_mean, 'i')
u_distance = gen.cosine_distance(u_mean, 'u')

'''
for i in a_distance:
    print(i)
print('-' * 30)
for i in i_distance:
    print(i)
print('-' * 30)
for i in u_distance:
    print(i)
print('-' * 30)
'''

print("Distance mean:")
print(f'a:{np.mean(a_distance)}')
print(f'i:{np.mean(i_distance)}')
print(f'u:{np.mean(u_distance)}')
print('-' * 30)

bins = 10
for index, distance  in enumerate([a_distance, i_distance, u_distance]):
    plt.figure()
    plt.hist(distance, bins=bins, density=False)
    plt.title(f'Distribution of distances for vowel {['a', 'i', 'u'][index]}')

print("A vs I:")
gen.stat_test(a_distance, i_distance)
print('-' * 30)

print("A vs U :")
gen.stat_test(a_distance, u_distance)
print('-' * 30 )

print("I vs U:")
gen.stat_test(i_distance, u_distance)
print('-' * 30)

print('\n')

# Entre diferentes vocales
ai_distance = gen.cosine_distance(a_mean, 'i')
au_distance = gen.cosine_distance(a_mean, 'u')

ia_distance = gen.cosine_distance(i_mean, 'a')
iu_distance = gen.cosine_distance(i_mean, 'u')

ua_distance = gen.cosine_distance(u_mean, 'a')
ui_distance = gen.cosine_distance(u_mean, 'i')

print("Distance mean:")
print(f'a:{np.mean(a_distance)}')
print(f'i:{np.mean(i_distance)}')
print(f'u:{np.mean(u_distance)}')
print('-' * 30)

for index, distance in enumerate([ai_distance, au_distance, ia_distance, iu_distance, ua_distance, ui_distance]):
    plt.figure()
    plt.hist(distance, bins=bins, density=False)
    plt.title(f'distances histogram for {['ai', 'au', 'ia', 'iu', 'ua', 'ui'][index]} pair of vowels')

print("A vs I:")
gen.stat_test(ai_distance, ia_distance)
print('-' * 30)

print("A vs U :")
gen.stat_test(au_distance, ua_distance)
print('-' * 30 )

print("I vs U:")
gen.stat_test(iu_distance, ui_distance)
print('-' * 30)

print('\n')

# Clasificacion ultimo grupo de audios
audio_real = [
        'A', 'A', 'A', 'A', 'A', 'A',
        'I', 'I', 'I', 'I', 'I', 'I',
        'U', 'U', 'U', 'U', 'U', 'U'
]

predicts = []
for letter in ['a', 'i', 'u']:
    predicts_vowel = gen.lastgroup(letter, a_mean, i_mean, u_mean)
    predicts.extend(predicts_vowel)

print(predicts)
print(audio_real)

labels = ['A','I','U']

cm = confusion_matrix(audio_real, predicts, labels=labels)

print(cm)

accuracy = accuracy_score(audio_real, predicts)

print(f"Aciertos: {accuracy*100:.2f}%")
print(f"Error: {(1-accuracy)*100:.2f}%")

plt.show()
