
class Building:

    def __init__(self, address, nb_floors):
        self.address = address
        self.nb_floors = nb_floors
        print("Bienvenue à la nouvelle ville et aux :", address, "/ ils sont au nombre de :", nb_floors)

# fatigue

building1 = Building("immeubles situé au 26 rue Granadine", 4)
building2 = Building("suermarche situé au 26 rue Granadine", 2)
building3 = Building("la bank situé au 26 rue Granadine", 1)





# TP : Simulateur de ville
# 1- Créer 3 classes : Immeuble(+nb_balcon), Suermarché(+ nb_rayons) et Banque (+ nb_coffre, name)
# 2- Créer Superclasse "Batiment" (add, nb_floors)
# 3- 4 immeubles, 2 supermarchés et 1 banques.