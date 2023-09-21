import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
from random import randrange
from PIL import ImageOps
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import tensorflow as tf
from skimage.io import imsave
from keras.models import model_from_json
from matplotlib import pyplot as plt
import cv2
import time 

class MyGUI:
    def __init__(self, master):
        # Initialize the GUI window
        self.master = master
        self.master.title("Colorizzz")
        self.master.configure(bg='#2b2d42')
        # Set the window size and aspect ratio
        self.width = 500
        self.height = 400
        self.aspect_ratio = 4/3
        self.master.geometry(f"{self.width}x{self.height}")
        self.master.resizable(False, False)

        # Create a canvas to display the image
        self.canvas1 = tk.Canvas(self.master, width=int(self.width/3), height=self.height, bd=0, highlightthickness=0, bg='#2b2d42')
        self.canvas1.pack(side="left", fill="both", expand=True)

        # Load and resize the image
        self.image_path1 = "cov.png"
        self.image1 = Image.open(self.image_path1)
        self.image1 = self.image1.resize((int(self.width/3), self.height))

        # Add the image to the canvas
        self.image_tk1 = ImageTk.PhotoImage(self.image1)
        self.canvas1.create_image(0, 0, anchor="nw", image=self.image_tk1)
        
        self.canvas = tk.Canvas(self.master, width=self.width/4, height=self.width/4, bd=0, highlightthickness=0, bg='#2b2d42')
        self.canvas.pack(side="left", fill="both", expand=True)

        # Load and resize the image
        self.image_path = "prev.png"
        self.image = Image.open(self.image_path)
        self.image = self.image.resize((int(self.width/4), int(self.width/4)))

        # Add the image to the canvas
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
        self.canvas.place(relx=0.65,rely=0.45,anchor="center")

        # Add some text inside the box
        self.text = tk.Label(self.master, text="Grayscale to Color", font=("Arial Bold", 22), bg='#2b2d42', fg='white')
        self.text.place(relx=0.65, rely=0.1, anchor="center")

        # Add a file browser button
        self.file_button = ttk.Button(self.master, text="Select file", command=self.select_file)
        self.file_button.place(relx=0.65, rely=0.8, anchor="center")
        
        self.start_button = ttk.Button(self.master, text="Start")
        self.start_button.place(relx=0.45, rely=0.7, anchor="center")
        
        self.save_button = ttk.Button(self.master, text="Save")
        self.save_button.place(relx=0.85, rely=0.7, anchor="center")

        # Add a folder browser button
        self.folder_button = ttk.Button(self.master, text="Select folder", command=self.select_folder)
        self.folder_button.place(relx=0.65, rely=0.9, anchor="center")

        # Add a catchphrase about color and black and white
        self.catchphrase = tk.Label(self.master, text="Transform your dull and lifeless grayscale images\ninto vibrant and colorful masterpieces with our color-ific\nGrayscale to Colourazitation algorithm!", font=("Arial Bold", 8), bg='#2b2d42', fg='grey')
        self.catchphrase.place(relx=0.65, rely=0.2, anchor="center")

    
    def filecolor(self,file_path):
            fol_name=os.path.dirname(file_path)
            if not os.path.exists(fol_name+'/result'):
                    os.makedirs(fol_name+'/result')
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            loaded_model.summary()
            loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
            fig, ax = plt.subplots(22, 2, figsize=(16, 100))
            row = 0
            
            colorize=[]
            colorize.append(img_to_array(load_img(file_path)))
            ax[row, 0].imshow(load_img(file_path), interpolation='nearest')
            colorize[0]=cv2.resize(colorize[0],(256,256))

            colorize = np.array(colorize, dtype=float)
            colorize = rgb2lab(1.0/255*colorize)[:,:,:,0]
            colorize = colorize.reshape(colorize.shape+(1,))

            # Test model
            output = loaded_model.predict(colorize)
            output = output * 128
            
            cur = np.zeros((256, 256, 3))
            cur[:, :, 0] = colorize[0][:, :, 0]
            cur[:, :, 1:] = output[0]
            resImage = lab2rgb(cur)
            ax[row, 1].imshow(resImage, interpolation='nearest')
            row += 1
            imsave('temp.jpg',resImage)
            self.preview_image('temp.jpg')     
            os.remove('temp.jpg')
            self.save_button.config(command=lambda:self.save_file_folder(resImage))

        
    def save_file_folder(self,resImage):
            fol_name = filedialog.askdirectory()
            
            imsave(fol_name+"/img_"+str(0)+".png", resImage)
            
       
            return
    def select_file(self):
        # Code to handle file selection
        file_path = filedialog.askopenfilename()
        if file_path:
            self.preview_image(file_path)
            self.start_button.config(command=lambda:self.filecolor(file_path))
            

    def update_canvas(self, folder_path):
        # get all images in folder
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]

        # create a 100x100 collage
        college = Image.new('RGB', (100, 100), color='white')
        x_offset = 0
        y_offset = 0

        # add images to collage
        num_images = min(len(images), 16)
        for img in images[:num_images]:
            try:
                image = Image.open(img).resize((25, 25))
                college.paste(image, (x_offset, y_offset))
            except:
                pass
            x_offset += 25
            if x_offset == 100:
                x_offset = 0
                y_offset += 25

        # add black/white tiles to collage
        for i in range(num_images, 16):
            color = (0+randrange(255), 0+randrange(255), 0+randrange(255)) if i % 2 == 0 else (255-randrange(255),255-randrange(255),255-randrange(255))
            college.paste(Image.new('RGB', (25, 25), color=color), (x_offset, y_offset))
            x_offset += 25
            if x_offset == 100:
                x_offset = 0
                y_offset += 25

        # convert to numpy array and stack images
        img_array = np.array(college)
        if img_array.shape == (100, 100, 3):
            img1 = np.vstack([np.hstack([img_array[0:50, 0:50], img_array[0:50, 50:100]]),
                              np.hstack([img_array[50:100, 0:50], img_array[50:100, 50:100]])])
        else:
            img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        fimg = Image.fromarray(img1)

        # update the image on the canvas
        self.image_tk = ImageTk.PhotoImage(fimg)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
    
    
    def fold_conv(self,folder_path):
        
            if not os.path.exists(folder_path+'/result'):
                    os.makedirs(folder_path+'/result')
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            loaded_model.summary()
            loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
            fig, ax = plt.subplots(22, 2, figsize=(16, 100))
            folder_items = os.listdir(folder_path)

            # Filter out only the image files
            image_files = [f for f in folder_items if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

            # Load and preprocess each image file
            colorize = []
            row = 0
            for filename in image_files:
                img = load_img(os.path.join(folder_path, filename))
                colorize.append(img_to_array(img))
                ax[row, 0].imshow(img, interpolation='nearest')
                row += 1
            colorize = np.array(colorize, dtype=float)
            colorize = rgb2lab(1.0/255*colorize)[:,:,:,0]
            colorize = colorize.reshape(colorize.shape+(1,))

            # Test model
            output = loaded_model.predict(colorize)
            output = output * 128

            row = 0
            
            for i in range(len(output)):
                cur = np.zeros((256, 256, 3))
                cur[:, :, 0] = colorize[i][:, :, 0]
                cur[:, :, 1:] = output[i]
                resImage = lab2rgb(cur)
                ax[row, 1].imshow(resImage, interpolation='nearest')
                row += 1

                imsave(folder_path+"/result/img_"+str(i)+".jpg", resImage)
                
#            os.remove(dirr)
            self.update_canvas(folder_path+"/result/")
    
        
    def select_folder(self):
        # Code to handle folder selection
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.update_canvas(folder_path)
            self.start_button.config(command=lambda:self.fold_conv(folder_path))
            
            

        
    def preview_image(self,image_path):
        # Code to preview the selected file or folder
        if not image_path:
            print("Select a file or folder first.")
            return

        try:
            image = Image.open(image_path)
            image = image.resize((150, 150))  # Resize the image to 100x100
            
            # Update the image on the canvas
            self.image_tk = ImageTk.PhotoImage(image)
            
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            return
        except Exception as e:
            print(f"Error loading image: {e}")







if __name__ == "__main__":
    root = tk.Tk()
    gui = MyGUI(root)
    root.mainloop()
