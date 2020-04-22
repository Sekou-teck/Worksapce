import string
from random import randint, choice
from tkinter import *

def generate_password():
    password_min = 6
    password_max = 12
    all_chars = string.ascii_letters + string.punctuation + string.digits
    password = "".join(choice(all_chars) for x in range(randint(password_min, password_max)))
    password_entry.delete(0, END)
    password_entry.insert(0, password)

# Creer la fenetre

window = Tk()
window.title("Générateur de Mot de pass")
window.geometry("720x480")
window.iconbitmap("password.ico")
window.config(background='#4065A4')

# Creer Frame
frame = Frame(window, bg='#4065A4')

# Creer une image
width = 300
height = 300
image = PhotoImage(file="password.png").zoom(35).subsample(32)
canvas = Canvas(frame, width=width, height=height, bg='#4065A4', bd=0, highlightthickness=0)
canvas.create_image(width/2, height/2, image=image)
# canvas.pack() : remplacer par grid pour cinder l'image et le titre du Mot de pass
canvas.grid(row=0, column=0, sticky=W)

# Creer une sous boite:
right_frame = Frame(frame, bg='#4065A4')

# Creer un titre
# label_title = Label(frame, text='Mot de pass', font=("Helvetica, 20"), bg="#4065A4", fg='white')
label_title = Label(right_frame, text="Mot de pass", font=("Helvetica", 20), bg='#4065A4', fg='white')
label_title.pack()

# Creer  une champ/entrée/input
password_entry = Entry(right_frame, font=("Helvetica", 20), bg='#4065A4', fg='white')
password_entry.pack()

# Creer un bouton
generate_password_button = Button(right_frame, text="générer", font=("Helvetica", 22), bg='#4065A4', fg='black', command= generate_password)
generate_password_button.pack(fill=X)

# on place la sous boite à droite de la frame principale
right_frame.grid(row=0, column=1, sticky=W)

# label_title.pack()
# label_title.grid(row=0, column=1, sticky=W)

# affiche frame
frame.pack(expand=YES)

# Creer une barre de menu
menu_bar = Menu(window)
# Creer un premier menu
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Nouveau", command=generate_password)
file_menu.add_command(label= "Quitter", command = window.quit)
menu_bar.add_cascade(label="Fichier", menu=file_menu)

# Configurer notre fenetre pour ajouter cette menu bar
window.config(menu=menu_bar)

# TP
# 1 - Creer une fenetre avec cookie au centre de l'écran
# 2 _ Créer un compteur
# 3 - Créer une boutique

# affiche la fenetre
window.mainloop()
