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
        return self.health

    def get_attack_value(self):
        return self.attack

    def damage(self, damage):
        self.health -= damage
   #     print("Aie ... Vous venez de subir", damage, "dégats !")

    def attack_player(self, target_player):
        target_player.damage(self.attack)


class Warrior(Player):

    def __init__(self, pseudo, health, attack):
        super().__init__(pseudo, health, attack)
        self.armor = 3
        print("Bienvenue au guerrier", pseudo, "/ point de vie:", health, "/ Attack:", attack)

    def damage(self, damage):
        if self.armor > 0:
            self.armor -= 1
            damage = 0
        super().damage(damage)

#        print("Aie ... Vous venez de subir", damage, "dégats !")

    def attack_player(self, target_player):
        target_player.damage(self.attack)

    def blade(self):
        self.armor = 3
        print("Les points d'armure ont été réchargés")

    def get_armor_point(self):
        return self.armor


player = Player("Sekou", 20, 3)
player.damage(3)
warrior = Warrior("DarkWarrior", 30, 4)

warrior.damage(4)
print("vie :", warrior.get_health(), "armor :", warrior.get_armor_point())
if issubclass(Warrior, Player):
    print("Le guerrier est bien une spécialisation de Player")

# TP : Simulateur de ville
# 1- Créer 3 classes : Immeuble, Suermarché et Banque
# 2- Créer Superclasse "Batiment"
# 3- 4 immeubles, 2 supermarchés et 1 banques.