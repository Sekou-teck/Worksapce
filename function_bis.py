# print, len, range, input
a = 0
while a < 12:
    a += 1
    print(a, a ** 2, a ** 3)

print(len("Je m'engage à respecter le confinement"))


def compteur3():
    i = 0
    while i < 3:
        print(i)
        i += 1


print("Bonjour ")
print(compteur3())


def double_compteur():
    compteur3()
    compteur3()


print(double_compteur())


def compteur(stop):
    i = 0
    while i < stop:
        print(i)
        i += 1


print(compteur(4))
print(compteur(2))


def compteur(stop):
    i = 0
    while i < stop:
        print(i)
        i += 1


a = 5
print(compteur(a))


def compteur_complet(start, stop, step):
    i = start
    while i < stop:
        print(i)
        i = i + step
print(compteur_complet(1, 7, ))