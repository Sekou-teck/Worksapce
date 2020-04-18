# Recolter une vcaleur nommée Porta-monnaie
#wallet = int(input("Entrer le nombre d'€ que vous possedez "))
#print("Vous avez actuellement, " wallet, " Euros")
# Créer un produit qui aura pour valeur 50€
#produit = 50
#print("Le produit vaut", produit, " Euros")
# Afficher la nouvelle valeur de Porte-monnaie, après l'achat
#wallet -= produit # || wallet = wallet - produit
#Affichage
#print("Produit acheté !")
#print("Il ne vous reste plus que", wallet, " Euros")
wallet = 5000
print("Vous avez actuellement", wallet, "€")
prix_ordinateur = 2500
print("Le produite vaut", prix_ordinateur, "€")
wallet -= prix_ordinateur
print("Produit acheté est Ordinateur !")
print("Il ne vous reste plus que", wallet, "€")
if prix_ordinateur<= 5000 and prix_ordinateur >= 1000:
    print("L'achat est possible")
else:
    print("L'achat n'est pas possible, vous n'avez que {}€".format(wallet))
    print(wallet)
# condition ternaire
veri = ("L'achat est possible", "L'achat est impossible")[prix_ordinateur <= 5000]
print(veri)


