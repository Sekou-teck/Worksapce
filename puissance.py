import puissance2 as pu2 # importer puissance2 en tant que module
b = 4
print("Le carre vaut", pu2.carre(b))

def carre(valeur):
    result = valeur ** 2
    return result


def cube(valeur):
    result = valeur ** 3
    return result
