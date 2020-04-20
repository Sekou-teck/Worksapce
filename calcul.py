def addition():
    result = 2020 - 1986
    return result


# Toutes ces 2 façons donnent le même résultat.
def multiply():
    return 5 * 8


def get_message():
    return "Le résultat du calcul est"


def addition2():
    return 5 + 4


def addition3():
    return 5 + 9


def addition4():
    return 5 + 15


print(get_message(), multiply())
print(get_message(), multiply())
print("Le résultat du calcul est", addition2())


# Utiliser un paramètre pour plusieurs opérations (n):
def addition(n = 56):
    return 5 + n


print("Le résultat du calcul est", addition())
print("Le résultat du calcul est", addition(9))
print("Le résultat du calcul est", addition(15))
