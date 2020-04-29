import numpy as np


def welcome():
    print("Bienvenue sur ma chaine Youtube")
    result = 87 + 87
    print("Le resultat du calcul est", result)


welcome()


def oiseau(voltage=100, eta="allumé", action="danse la Java"):
    print("Le pirroquet ne pourra pas ", action)
    print("Si vous le branchez sur ", voltage, "volts !")
    print("L'auteur de ceci est complètement", eta)


oiseau(voltage=250, eta="givré", action="vous approuver")


def cube(n):
    return n ** 3


def volume_sphere(r):
    return 4 / 3 * np.pi * cube(r)


r = float(input("Entre une valeur du rayon :"))
print("Le volume de cette sphere vaut", volume_sphere(r))
