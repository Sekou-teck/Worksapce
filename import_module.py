from puissance import carre # importe une seule fonction
from puissance import carre, cube # imoprte explicitement 2 fonctions
from puissance import * # importe toute les fonctions
import puissance # importe le module
import puissance as pu # importe un module et lui donne un alias
from puissance import carre as ca # importe une fonction d'un module et lui donne un alias


a = 5
u = carre(a)
print("Le carre vaut", u)
v = cube(a)
print("Le cube vaut", v)
