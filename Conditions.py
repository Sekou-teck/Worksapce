def main():
    #print("Hello Sekouba")
    #print("Bonjour à tout le monde ! Notre premier pas professionnel vers Py")
    # création de variable
    username = "Sekouba"
    age = 34
    Gender = 'Male'
    isHappy = True
    # affichage des variables
    #modification de la valeur de la variable age
    age = 28
    age = age + 6
    #Concatenation dans des variables
    print("Bonjour " + username + ", Vous venez d'avoir " + str(age) + " ans ! ")
    # Recolter une primière note
    note1 = int(input("Entrer la premiere note"))
    #Recolter la seconde note
    note2 = int(input("Entrer la seconde note"))
    #Reconter la troisième note
    note3 = int(input("Entrer la troisième note"))
    #Calculer la moyenne
    resultat =(note1 + note2 + note3)/3
    print("Le resultat est " + str(resultat))

    # Recolter une vcaleur nommée Porta-monnaie
    # Créer un produit qui aura pour valeur 50€
    # Afficher la nouvelle valeur de Porte-monnaie, après l'achat
    porteMonnaie = 15000
    prixOrdinateur = 2500
    if prixOrdinateur <= 2000:
        print("Le prix de l'ordinateur est bien inférieur ")
    else:
        print("Le prix de l'ordinateur est supérieur !")


if __name__ == '__main__':
    main()