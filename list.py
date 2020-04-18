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
