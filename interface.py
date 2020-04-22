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
