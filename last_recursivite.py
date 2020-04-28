
def add(a):
    a += 1
    print(a)
    if a < 15:
        add(a)


add(2)

#TP : une fonction pour calculer le nombre de voyelles dans un mots
# - définir une fonction get_vowels_numbers(mot)
# - créer un compteur de voyelles
# - pour chaque lettre du mot, il faut vérifier s'il s'agit d'un voyelle
# - SI c'est le cas, on ajoute 1 au compteur
# - à la fin de la fonction, il faut renvoyer le compteur.