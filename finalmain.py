import Codigo.functions.general as gen
import numpy as np
import scipy.spatial as spp
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu






def firstgroup(letra):
    vowel = []
    for genero in ['Hombre', 'Mujer']:
        for i in range(1, 10 + 1):
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            vowel.append(gen.process_audio(audio_file))
    vowel = np.asarray(vowel)
    vowel = np.mean(vowel, axis=0)
    return vowel

def cosine_distance(vowel_mean, letra):
    distance = []
    for genero in ['Hombre', 'Mujer']:
        for i in range(11, 15 + 1):        
            audio_file = f'Audios/{genero}/{letra}/{letra}_{i}.wav'
            vowel_rep=gen.process_audio(audio_file)
            distance.append(spp.distance.cosine(vowel_rep,vowel_mean))
    return distance


def distance_tests(vowel1, vowel2, alpha=0.05):
    test=shapiro(vowel1)
    p1=test.pvalue
    test=shapiro(vowel2)
    p2=test.pvalue
    if (p1>alpha and p2>alpha):
        print("both follow a normal distribution")
        test=ttest_ind(vowel1,vowel2)
        pval=test.pvalue
        if(pval>alpha):
            print("Both mean are the same")
        else:
            print("Means are different")
    else:
        print("At least one does not follow a normal distribution")
        test=mannwhitneyu(vowel1,vowel2)
        pval=test.pvalue
        if (pval>alpha):
            print("Distributions are similars")
        else:
            print("Distributions are different")



a_mean = firstgroup('a')
i_mean = firstgroup('i')
u_mean = firstgroup('u')

a_distance = cosine_distance(a_mean, 'a')
i_distance = cosine_distance(i_mean, 'i')
u_distance = cosine_distance(u_mean, 'u')

for i in a_distance:
    print(i)
print('-' * 30)
for i in i_distance:
    print(i)

print('-' * 30)
for i in u_distance:
    print(i)
print('-' * 30)

print("distance mean:")
print(f'a:{np.mean(a_distance)}')
print(f'i:{np.mean(i_distance)}')
print(f'u:{np.mean(u_distance)}')
print('-' * 30)




for letter,i  in enumerate([a_distance, i_distance, u_distance]):
    plt.figure()
    plt.hist(i, bins=5, density=False)
    if (letter==0):
        plt.title('distances histogram for a vowel')
    if (letter==1):
        plt.title('distances histogram for i vowel')
    if (letter==2):
        plt.title('distances histogram for u vowel')
plt.show()

print("A vs I:")
distance_tests(a_distance,i_distance)
print('-' * 30)

print("A vs U :")
distance_tests(a_distance,u_distance)
print('-' * 30 )


print("I vs U:")
distance_tests(i_distance,u_distance)
print('-' * 30)
