from tkinter import *
import webbrowser


def open_sekou_channel():
    webbrowser.open_new("https://www.youtube.com/channel/UCGwnbXmThEOMwLBLf-5FMDQ?view_as=subscriber")


# créer la première fenêtre :

window = Tk()

# Personnaliser ma fenêtre

window.title("My Application")
window.geometry("720x480")
window.minsize(400, 360)
window.iconbitmap("IMG_2845.ico")
window.config(background='#41B77f')

# Créer une frame (boite) dans laquelle seront stockés les textes et couleurs.
frame = Frame(window,
              bg="#41B77f")  # , bd=1, relief= SUNKEN) # bd=1 et relief = SUNKEN, sont là pour la bordure des textes.

# Ajouter un premier texte
label_title = Label(frame, text="Bienvenue sur l'Application", font=("Courrier", 50), bg="#41B77f", fg='White')
label_title.pack()

# ajouter un second texte (sous-titre)
label_subtitle = Label(frame, text="Salut tous, c'est Sekouba", font=("Courrier", 25), bg="#41B77f", fg='White')
label_subtitle.pack()

# Ajouter un premier bouton
yt_boutton = Button(frame, text="Ouvrir Youtube", font=("Courrier", 25), fg='#41B77f', bg="white",
                    command=open_sekou_channel)
yt_boutton.pack(pady=25, fill=X)

# ajouter l'extension de la frame
frame.pack(expand=YES)

# afficher la fenêtre

window.mainloop()
# Corrigé TP

# 1-
from tkinter import *

cookie_count = 0


def add_cookie():
    global cookie_count
    cookie_count += 1
    label_counter.config(text=cookie_count)


# creer la fenetre
window = Tk()
window.title("Cookie Clicker")
window.geometry("720x480")
window.iconbitmap("cookie.ico")
window.config(background='#dee5dc')

# ajout du compteur
label_counter = Label(window, text="0", font=("Courrier", 30), bg="#dee5dc")
label_counter.pack()

# creer la frame principale
frame = Frame(window, bg='#dee5dc')

# creation d'image
width = 300
height = 300
image = PhotoImage(file="cookie.png").zoom(32).subsample(64)

# ajout du bouton/image
button = Button(frame, image=image, bg='#dee5dc', bd=0, relief=SUNKEN, command=add_cookie)
button.pack()

# ajout de la frame au centre
frame.pack(expand=YES)

# affichage
window.mainloop()