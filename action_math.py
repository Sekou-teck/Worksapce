# All to one
import numpy as np

n = range(3, 15, 2)
print(n)
print(type(n))

u = [20, 65, 32]
print(type(u))
print(list(range(3, 15, 2)))

np.arange(0, 11 * np.pi, np.pi)

np.linspace(3, 9, 10)



Fonctions trigonométriques
numpy.sin(x) 	sinus
numpy.cos(x) 	cosinus
numpy.tan(x) 	tangente
numpy.arcsin(x) 	arcsinus
numpy.arccos(x) 	arccosinus
numpy.arctan(x) 	arctangente

Fonctions hyperboliques

numpy.sinh(x) 	sinus hyperbolique
numpy.cosh(x) 	cosinus hyperbolique
numpy.tanh(x) 	tangente hyperbolique
numpy.arcsinh(x) 	arcsinus hyperbolique
numpy.arccosh(x) 	arccosinus hyperbolique
numpy.arctanh(x) 	arctangente hyperbolique

Fonctions diverses

x**n 	x à la puissance n, exemple : x**2
numpy.sqrt(x) 	racine carrée
numpy.exp(x) 	exponentielle
numpy.log(x) 	logarithme népérien
numpy.abs(x) 	valeur absolue
numpy.sign(x) 	signe

Fonctions utiles pour les nombres complexes

numpy.real(x) 	partie réelle
numpy.imag(x) 	partie imaginaire
numpy.abs(x) 	module
numpy.angle(x) 	argument en radians
numpy.conj(x) 	complexe conjugué


Comparaison entre around(x,0) et trunc(x)

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 500)
plt.plot(x, np.around(x,0), label="around(x,0)")
plt.plot(x, np.trunc(x), label="trunc(x)")
plt.legend()

plt.show()

Comparaison entre around(x,0) et trunc(x)

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 500)
plt.plot(x, np.around(x,0), label="around(x,0)")
plt.plot(x, np.trunc(x), label="trunc(x)")
plt.legend()

plt.show()

# Courbe :

#Syntaxe « PyLab »

from pylab import *

x = array([1, 3, 4, 6])
y = array([2, 3, 5, 1])
plot(x, y)

show() # affiche la figure a l'ecran

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 4, 6])
y = np.array([2, 3, 5, 1])
plt.plot(x, y)

plt.show() # affiche la figure a l'ecran


Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y = np.cos(x)
plt.plot(x, y)

plt.show() # affiche la figure a l'ecran

#Syntaxe « PyLab »

from pylab import *

x = linspace(0, 2*pi, 30)
y = cos(x)
plot(x, y)
title("Fonction cosinus")

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y = np.cos(x)
plt.plot(x, y)
plt.title("Fonction cosinus")

plt.show()

# Labels sur axe Xlabels et Ylabels

from pylab import *

x = linspace(0, 2*pi, 30)
y = cos(x)
plot(x, y)
xlabel("abscisses")
ylabel("ordonnees")

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y = np.cos(x)
plt.plot(x, y)
plt.xlabel("abscisses")
plt.ylabel("ordonnees")

plt.show()

# Affichage de plusieurs courbes :

from pylab import *

x = linspace(0, 2*pi, 30)
y1 = cos(x)
y2 = sin(x)
plot(x, y1)
plot(x, y2)

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y1 = np.cos(x)
y2 = np.sin(x)
plt.plot(x, y1)
plt.plot(x, y2)

plt.show()

# Avec un legende :

#Syntaxe « PyLab »

from pylab import *

x = linspace(0, 2*pi, 30)
y1 = cos(x)
y2 = sin(x)
plot(x, y1, label="cos(x)")
plot(x, y2, label="sin(x)")
legend()
show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y1 = np.cos(x)
y2 = np.sin(x)
plt.plot(x, y1, label="cos(x)")
plt.plot(x, y2, label="sin(x)")
plt.legend()
plt.show()

#Format de courbes :

Syntaxe « PyLab »

from pylab import *

x = linspace(0, 2*pi, 30)
y1 = cos(x)
y2 = sin(x)
plot(x, y1, "r--", label="cos(x)")
plot(x, y2, "b:o", label="sin(x)")
legend()

show()

Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y1 = np.cos(x)
y2 = np.sin(x)
plt.plot(x, y1, "r--", label="cos(x)")
plt.plot(x, y2, "b:o", label="sin(x)")
plt.legend()

plt.show()

#Style de ligne

#Les chaînes de caractères suivantes permettent de définir le style de ligne :
#Chaîne 	Effet
# - 	ligne continue
# -- 	tirets
: 	ligne en pointillé
-. 	tirets points

# Syntaxe « PyLab »
#
# from pylab import *
#
# x = linspace(0, 2*pi, 20)
# y = sin(x)
# plot(x, y, "o-", label="ligne -")
# plot(x, y-0.5, "o--", label="ligne --")
# plot(x, y-1, "o:", label="ligne :")
# plot(x, y-1.5, "o-.", label="ligne -.")
# plot(x, y-2, "o", label="pas de ligne")
# legend()
#
# show()
#
# Syntaxe « standard »
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(0, 2*np.pi, 20)
# y = np.sin(x)
# plt.plot(x, y, "o-", label="ligne -")
# plt.plot(x, y-0.5, "o--", label="ligne --")
# plt.plot(x, y-1, "o:", label="ligne :")
# plt.plot(x, y-1.5, "o-.", label="ligne -.")
# plt.plot(x, y-2, "o", label="pas de ligne")
# plt.legend()
#
# plt.show()

# Symboles <Marker>

Symbole (« marker »)

# Les chaînes de caractères suivantes permettent de définir le symbole (« marker ») :
# Chaîne 	Effet
. 	point marker
, 	pixel marker
o 	circle marker
v 	triangle_down marker
^ 	triangle_up marker
< 	triangle_left marker
> 	triangle_right marker
1 	tri_down marker
2 	tri_up marker
3 	tri_left marker
4 	tri_right marker
s 	square marker
p 	pentagon marker
* 	star marker
h 	hexagon1 marker
H 	hexagon2 marker
+ 	plus marker
x 	x marker
D 	diamond marker
d 	thin_diamond marker
| 	vline marker
_ 	hline marker

# Couleurs :

Chaîne 	Couleur en anglais 	Couleur en français
b 	blue 	bleu
g 	green 	vert
r 	red 	rouge
c 	cyan 	cyan
m 	magenta 	magenta
y 	yellow 	jaune
k 	black 	noir
w 	white 	blanc

# Largeur des lignes :

from pylab import *

x = linspace(0, 2*pi, 30)
y1 = cos(x)
y2 = sin(x)
plot(x, y1, label="cos(x)")
plot(x, y2, label="sin(x)", linewidth=4)
legend()

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 30)
y1 = np.cos(x)
y2 = np.sin(x)
plt.plot(x, y1, label="cos(x)")
plt.plot(x, y2, label="sin(x)", linewidth=4)
plt.legend()

plt.show()


# Tracé des formes :

from pylab import *

x = array([0, 1, 1, 0, 0])
y = array([0, 0, 1, 1, 0])
plot(x, y)
xlim(-1, 2)
ylim(-1, 2)

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 1, 0, 0])
y = np.array([0, 0, 1, 1, 0])
plt.plot(x, y)
plt.xlim(-1, 2)
plt.ylim(-1, 2)

plt.show()

# instruction 'axis'(egal):

#Syntaxe « PyLab »

from pylab import *

x = array([0, 1, 1, 0, 0])
y = array([0, 0, 1, 1, 0])
plot(x, y)
axis("equal")

show()

#Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 1, 0, 0])
y = np.array([0, 0, 1, 1, 0])
plt.plot(x, y)
plt.axis("equal")

plt.show()


Syntaxe « PyLab »

from pylab import *

x = array([0, 1, 1, 0, 0])
y = array([0, 0, 1, 1, 0])
plot(x, y)
axis("equal")
xlim(-1, 2)

show()

Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 1, 0, 0])
y = np.array([0, 0, 1, 1, 0])
plt.plot(x, y)
plt.axis("equal")
plt.xlim(-1, 2)

plt.show()

# Tracé un cercle :

Syntaxe « PyLab »

from pylab import *

theta = linspace(0, 2*pi, 40)

x = cos(theta)
y = sin(theta)
plot(x, y)

show()

Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 40)

x = np.cos(theta)
y = np.sin(theta)
plt.plot(x, y)

plt.show()

# Cercle avec l'instruction "egal":

Syntaxe « PyLab »

from pylab import *

theta = linspace(0, 2*pi, 40)

x = cos(theta)
y = sin(theta)
plot(x, y)
axis("equal")

show()

Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 40)

x = np.cos(theta)
y = np.sin(theta)
plt.plot(x, y)
plt.axis("equal")

plt.show()

# Possible de modifier le domaine de l'abscisse :

Syntaxe « PyLab »

from pylab import *

theta = linspace(0, 2*pi, 40)

x = cos(theta)
y = sin(theta)
plot(x, y)
axis("equal")
xlim(-3,3)

show()

Syntaxe « standard »

import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 40)

x = np.cos(theta)
y = np.sin(theta)
plt.plot(x, y)
plt.axis("equal")
plt.xlim(-3,3)

plt.show()

# TP :
# 1 -
# integration numerique par la methode des rectangles avec alpha = a

import numpy as np
import matplotlib.pyplot as plt

xmin = 0
xmax = 3*np.pi/2
nbx = 20
nbi = nbx - 1 # nombre d'intervalles

x = np.linspace(xmin, xmax, nbx)
y = np.cos(x)
plt.plot(x,y,"bo-")

integrale = 0
for i in range(nbi):
    integrale = integrale + y[i]*(x[i+1]-x[i])
    # dessin du rectangle
    x_rect = [x[i], x[i], x[i+1], x[i+1], x[i]] # abscisses des sommets
    y_rect = [0   , y[i], y[i]  , 0     , 0   ] # ordonnees des sommets
    plt.plot(x_rect, y_rect,"r")
print("integrale =", integrale)

plt.show()


# Polynome de Lagrange :

from pylab import *

a = 0
b = 8
m = (a+b)/2

x = linspace(a, b, 101)

l0 = (x-m)/(a-m)*(x-b)/(a-b)
l1 = (x-a)/(m-a)*(x-b)/(m-b)
l2 = (x-a)/(b-a)*(x-m)/(b-m)

plot([a,m,b],zeros(3),"s") # position des valeurs 0
plot([a,m,b],ones(3), "o")  # position des valeurs 1
plot(x,l0, label="l0")
plot(x,l1, label="l1")
plot(x,l2, label="l2")

title("Polynomes de Lagrange")
xlim(-1,9)
ylim(-1,2)
text(a,-0.1,"(a,0)",ha="center",va="top")
text(m,-0.1,"(m,0)",ha="center",va="top")
text(b,-0.1,"(b,0)",ha="center",va="top")
text(a,1.05,"(a,1)",ha="center",va="bottom")
text(m,1.05,"(m,1)",ha="center",va="bottom")
text(b,1.05,"(b,1)",ha="center",va="bottom")
legend()
grid()

show()

# Extrapolation de Lagrange avec un polynome de degre 2

from pylab import *

a = 0
b = 8
m = (a+b)/2
# valeurs de la fonction en a, m, et b
ya = 4
ym = 8
yb = -6

x = linspace(a, b, 101)

l0 = (x-m)/(a-m)*(x-b)/(a-b)
l1 = (x-a)/(m-a)*(x-b)/(m-b)
l2 = (x-a)/(b-a)*(x-m)/(b-m)
P = ya*l0 + ym*l1 + yb*l2

plot([a,m,b],zeros(3),"s") # position des valeurs 0
plot([a,m,b],[ya,ym,yb], "o")  # position des valeurs de f
plot(x,P)

title("Interpolation de Lagrange")
xlim(-1,9)
ylim(-8,10)
text(a,-0.5,"(a,0)",ha="center",va="top")
text(m,-0.5,"(m,0)",ha="center",va="top")
text(b,-0.5,"(b,0)",ha="center",va="top")
text(a,ya+0.05,"(a,f(a))",ha="right",va="bottom")
text(m,ym+0.2,"(m,f(m))",ha="left",va="bottom")
text(b,yb-0.4,"(b,f(b))",ha="left",va="top")
grid()

show()
