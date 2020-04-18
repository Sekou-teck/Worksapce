
#Vérification de Mot de pass
password = input("Entrer votre mot de pass")
password_length = len(password)
print(password_length)
if password_length <= 10:
    print("Mot de pass trop court")
elif 8 < password_length < 12:
    print('Mot de pass moyen')
else:
    print("Mot de pass parfait !")
#TP: place de cinéma :
# 1 - recolter l'âge de la personne (Quel est votre age ?)
# SI la personne est mineur, elle paye 7€ SINON, elle payera 12€
# Souhaiteriez-vous Pop-corn ?
# SI oui, 5€ supplémentaire SINON, prix d'une place
# Afficher prix total à payer

# Résolution TP:

