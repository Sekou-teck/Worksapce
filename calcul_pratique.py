# créer une fonction max() qui va renvoyer le résultat le plus haut parmis 2 valeurs:
def max(a, b):
    if a > b:
        return a
    else:
        return b


first_value = int(input("Entrer la valeur de a (Première valeur)"))
second_value = int(input("Entre la valeur de b (second valeur)"))
max_value = max(first_value, second_value)
print("La valeur max est ", max(first_value, second_value)) # || print("La valeur max est ", max())



