import statistics
from random import shuffle
list = [
    20, 36, 24, 54, 76,
    28, 98, 67, 54, 23,
    1, 8, 19, 43, 36, 67
]
list1 = ["Sekouba", "Aissata", "Aicaley", "Ibrahim"]
print(list1[0])
print(list1[:])
print(list1[len(list1)-1])
list1[0] = "Sekou"
list1[2:4] = ["Aica", "Ibra"]
print(list1)
#Ajout d'un élément sur la liste :
list1.append("Lamine")
print(list1)
#ajout plusieurs éléments :
list1.extend(["Mariama", "Moussa"])
print(list1)
#Supprimer un éléments de la liste :
del list1[-1]
print(list1)
print(len(list1))
list1.pop(-1)
print(list1)
list1.remove("Lamine")
print(list1)
list1[2:4] = ["Aicaley", "Ibrahim"]
print(list1)
list1[0] = ("Sekouba")
print(list1)
# del = .remove = .pop
# Vider toute la liste :
list1.clear()
print(list1)
# .clear = del list[:]
result = statistics.mean(list)
result = statistics.geometric_mean(list)
print(result)
print("La moyenne de la liste {}".format(list))
# Mélanger les éléments de la liste à l'affichage
shuffle(list)
print(list)
#découper les éléments par un délimiteur
text = input("Entrer une chaine de la forme(email_Pseudo_password)").split("_")
print(text)
print("Salut {}, ton email {}, ton mot de pass {}".format(text[1], text[0], text[2]))

# Corrigé TP

from random import shuffle

# Générateur de phrases
# demander en console une chaine de la forme "mot1/mot2/mot3/mot4"
chained_words = input("Entrer une chaine de la forme mot1/mot2/mot3/mot4")

# transformer cette chaine en liste
words = chained_words.split("/")

# la melanger
shuffle(words)

# recuperer le nombre d'elements
words_len = len(words)

# si le nombre d'élements de cette liste est inferieur à 10
if words_len < 10:
    # afficher les deux premiers mots
    print(words[0], words[1])
    # ou
    print(words[0:1])
# si le nombre d'éements est superieur ou égal à 10
else:
    # afficher les 3 derniers
    print(words[words_len - 1], words[words_len - 2], words[words_len - 3])
    # ou
    last_value = words[words_len] - 1
    pre_last_value = last_value - 2
    print(words[pre_last_value:last_value])
