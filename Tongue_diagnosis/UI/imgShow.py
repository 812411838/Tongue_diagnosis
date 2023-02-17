from tkinter import *
from PIL import Image, ImageTk

import tkinter
from PIL import Image, ImageTk
root = tkinter.Tk()
img_open = Image.open("img/4.png")
img_png = ImageTk.PhotoImage(img_open)
label_img = tkinter.Label(root, image = img_png)
label_img.pack()
root.mainloop()


