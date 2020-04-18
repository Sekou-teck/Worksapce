print("Vous êtes le client n°1")
print("Vous êtes le client n°2")
print("Vous êtes le client n°3")
print("Vous êtes le client n°4")
print("Vous êtes le client n°5")
# for : pour une valeur de départ (1), jusqu'à une valeur d'arrivée (5)
for num_client in range (1, 6):
    print("Vous êtes le client n°", num_client)
# for each : pour chaque valeur d'une liste donnée
print("Email envoyé à courtiers@yt.fr")
print("Email envoyé à contact@yt.fr")
print("Email envoyé à burite@yt.fr")
Emails = ["courtiers@yt.fr", "contact@yt.fr", "burite@yt.fr", "sfofana@gmail.com", "sfn@sfr.fr", "abcd@yahoo.fr"]
blacklist = ["sfofana@gmail.com", "sfn@sfr.fr", "abcd@yahoo.fr", "contact@yt.fr"]
for email in Emails:
    if email in blacklist:
        print("Email {} interdit ! Envoi impossible ...".format(email))
        continue
        # pour continuer jusq'au bout de la boucle ou Breack pour arrêter l'exécution une fois qu'une condition est rencontrée
        print('Email envoyé à', email)
        # boucle while : tant qu'une condition est vraie
salaire = 1500
while salaire < 10000:
    salaire = salaire + 120
    print("Votre salaire actuel est de ", salaire, "€")
print("Fin de programme")
# un Youtubeur à des 2500 abonnés et il gagne 10% d'audience par mois. combien d'abonné a-t-il en 2 ans ?
subcriber_count = 2500
month = 0
while month <= 24:
    subcriber_count *= 1.10
    print("Vous avez actuellement {} abonnés !".format(subcriber_count))
    month +=1 # Cela implique que l'on passe au mois suivant SINON, le boucle s'exécute à l'infini.
TP : Jeu de juste prix
# Choisir un nombre entre 1 et 1000
# Tant que le jeu n'est pas fini
# --> demander à l'utilisateur "entrer un prix"
# --> Si il trouve le juste prix, "C'est gagné !"
# --> Sinon, on affiche "C'est moins" ou  "C'est plus"