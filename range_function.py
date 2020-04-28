# Types int, float, str, complex, tuples
a = True
b = not a
c = 1 + 3J
d = (2, 3, 5, 8, 6)
(e, f) = (3, '28a')
print(type(f))


# Tuples
c = 3 + 4j

#def test():
#    return 2, 5
# dictionnaire
mon_dictionnaire = {"voiture": "véhicule de quatre roues", "vélo": "véhicule de deux roues"}
mon_dictionnaire["tricycles"]= "véhicule à trois roues"
nb_pneus = {}
nb_pneus["voiture"] = 4
nb_pneus["vélo"]= 2
nb_pneus["tricycles"] = 3
print(nb_pneus)
print(type(mon_dictionnaire))

for i in nb_pneus.items():
    print(i)

for cle, valeur in nb_pneus.items():
    print("L'élement de clé est ", cle, "vaut ", valeur)
