from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
from pymsgbox import *


root=tk.Tk()

root.title("Depression analysis using Face Recognition System")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

image2 =Image.open('images/f3.jpg')
image2 =image2.resize((1400,700))

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

# w = tk.Label(root, text="Facial Expression using Face Recognition System",width=40,background="skyblue",height=2,font=("Times new roman",19,"bold"))
# w.place(x=0,y=15)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="skyblue")


from tkinter import messagebox as ms


def Login():
    from subprocess import call
    call(["python","login.py"])
def Register():
    from subprocess import call
    call(["python","registration.py"])


wlcm=tk.Label(root,text="Prediction of Human Psychological State by Analyzing the Facial Features",width=100,height=1,background="skyblue",foreground="black",font=("Times new roman",18,"bold"))
wlcm.place(x=0,y=450)


d2=tk.Button(root,text="Login",command=Login,width=14,height=1,bd=0,background="blue",foreground="white",font=("times new roman",18,"bold"))
d2.place(x=750,y=50)


d3=tk.Button(root,text="Sign-Up",command=Register,width=14,height=1,bd=0,background="green",foreground="white",font=("times new roman",18,"bold"))
d3.place(x=1000,y=50)



root.mainloop()
