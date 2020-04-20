
from model.player import Player
from model.weapon import Weapon

knife = Weapon("Couteau", 3)
# POur un objet, on crée une classe dans laquelle on ajoute des attributs
# Un constructeur pour initialiser les caractéristique de chaque objet (__init__(self):)
# Il faut des méthodes pour bien mouliner nos fonctions dans cette classe

player1 = Player("Sekou", 20, 3)
player1.damage(3)
print("vous possedez désormais", player1.get_health(), "Point de vie")
print("Pseudo:", player1.get_pseudo())
print("health: ", player1.get_health())
print("Attack: ", player1.get_attack())

#print("Bienvenue au joueur", player1.pseudo)

player2 = Player("Aissata", 20, 5)
#print("Bienvenue au joueur", player2.pseudo)
player1.attack_player(player2)
print(player1.get_pseudo(), "attaque", player2.get_pseudo())
print("Bienvenue au joueur", player1.get_pseudo(), "/ point de vie:", player1.get_health(), "/ Attack:", player1.get_attack())
print("Bienvenue au joueur", player2.get_pseudo(), "/ point de vie:", player2.get_health(), "/ Attack:", player2.get_attack())