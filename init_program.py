import os
import tkinter as tk
from tkinter import filedialog, messagebox
from test import *



class Main:
    def __init__(self, master):
        root.title("Picture To Excel File")
        self.master = master

        # GUI components
        self.image_label = tk.Label(self.master, text="Image Path")
        self.image_label_entry = tk.Entry(self.master, validate="focusout")
        self.image_browse_button = tk.Button(self.master, text="Browse",
                                              command=lambda: self.update("Browser", self.master))

        self.output_folder = tk.Label(self.master, text="Output Folder Path")
        self.output_folder_entry = tk.Entry(self.master, validate="focusout")
        self.output_folder_browse_button = tk.Button(self.master, text="Browse",
                                                   command=lambda: self.update("Output", self.master))

        self.convert_button = tk.Button(self.master, text="Convert",
                                        command=lambda: self.update("Convert", self.master))

        # GUI layout
        self.image_label.grid(row=2, column=20, padx=5, pady=5)
        self.image_label_entry.grid(row=2, column=23, columnspan=3, padx=5, pady=5)
        self.image_browse_button.grid(row=2, column=26, padx=5, pady=5)
        self.output_folder.grid(row=3, column=20, padx=5, pady=5)
        self.output_folder_entry.grid(row=3, column=23, columnspan=3, padx=5, pady=5)
        self.output_folder_browse_button.grid(row=3, column=26, padx=5, pady=5)
        self.convert_button.grid(row=10, column=19, columnspan=10, padx=5, pady=5)
        self.convert_button.config(height=2, width=15)
        self.convert_button['state']

    def update(self, method, master):

        if method == "Browser":
            imagePath = filedialog.askopenfilename()
            self.image_label_entry.delete(0, tk.END)
            self.image_label_entry.insert(0, imagePath)

        if method == "Output":
            outputPath = filedialog.askdirectory()
            self.output_folder_entry.delete(0, tk.END)
            self.output_folder_entry.insert(0, outputPath)

        if method == "Convert":
            self.stringImage = self.image_label_entry.get()
            self.stringOutput = self.output_folder_entry.get()


            if self.stringImage=="":
                messagebox.showerror("Missing Data", "Please fill the image path, and then press on the 'Convert' button")

            elif not os.path.exists(self.stringImage):
                messagebox.showerror("Path Validation", "Image Path is not a valid path")

            elif self.stringOutput=="":
                messagebox.showerror("Missing Data",
                                     "Please fill the folder path, and then press on the 'Convert' button")

            elif not os.path.exists(self.stringOutput):
                messagebox.showerror("Path Validation", "Folder Path is not a valid path")

            else:
                convert(self.stringImage, self.stringOutput)
                messagebox.showinfo("File was saved", "New file was saved!")

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("370x200")
    root.resizable(0, 0)
    my_gui = Main(root)
    root.mainloop()

