
class Player:

    def __init__(self, pseudo, health, attack):
        self.pseudo = pseudo
        self.health = health
        self.attack = attack
        self.weapon = None
        print("Bienvenue au joueur", pseudo, "/ point de vie:", health, "/ Attack:", attack)

    def get_pseudo(self):
        return self.pseudo
    def get_health(self):
        return  self.health
    def get_attack(self):
        return self.attack

    def damage(self, damage):
        self.health -= damage
        print("Aie ... Vous venez de subir", damage, "dégats !")

    def attack_player(self, target_player):
        target_player.damage(self.attack)