# Copy un fichier d'un endroit à un autre
import os
import shutil

# 1ere méthode de création de fichier
open("students.txt", "w+")
file = open("students", "w+")
file.write("Sekouba\n")
file.write("Aissata\n")
file.write("Aicaley\n")
file.write("Ibrahim\n")
file.close()

#2ème méthode :
with open("students.txt", "a+") as file:
    file.write("Sekouba\n")
    file.write("Aissata\n")
    file.write("Aicaley\n")
    file.write("Ibrahim\n")
    file.write("Haî\n")
    file.write("Mariama\n")
    file.write("Ibrahima\n")
    file.close()

#3ème méthode
students_list = ["Sekouba", "Aissata", "Sanfan", "Aicaley", "Ibrahim", "Haî", "Mariama", "Ibrahima"]
with open("students.txt", "w+") as file:
    for student in students_list:
        file.write(student + "\n")
    file.close()

# 1 - Lecture

import os

if os.path.exists("mealss.txt"):
    with open("mealss.txt", "r+") as file:
        print(file.readlines())
        file.close()
else:
    print("Le document n'existe pas ! Attention !")

# 2 - Lecture

import os
import random

if os.path.exists("meals.txt"):
    with open("meals.txt", "r+") as file:
        meals_list = file.readlines()
        meal_random_choise = random.choice(meals_list)
        print("Je vous propose aujourd'hui, le répas", meal_random_choise)
        file.close()
else:
    print("Le document n'existe pas ! Attention !")

import os
import shutil


source = "IMG_2845.ico"
target = "images/IMG_2845.ico"

shutil.copy(source, target)
os.remove(source)

#TP:
# 1- Depuis le générateur de mot de pass de la dernière fois, créer un système de stockage de mot de pass
# 2- Créer une interface graphique pour le générateur de repas (lecture + interface graphique)