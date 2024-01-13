from tkinter import ttk
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk

# constante
MODEL_BOX = 1
MODEL_SEG = 2

class Toolbar:
    """Barre d'outils à ajouter à une fenêtre"""

    def __init__(self, root, exit_function=None):
        self.root = root

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # création de la barre de menu
        self.toolbar = tk.Menu(self.root)
        self.root.configure(menu=self.toolbar)  # affichage du Menu

        # création du menu défilant
        self.drop_down_menu = tk.Menu(self.toolbar, tearoff=0,
                                      font=("Calibri", 10))
        self.drop_down_menu.add_command(label="Nouveau", command=None,
                                        state=tk.DISABLED)
        self.drop_down_menu.add_command(label="Importer", command=self.importation)
        self.drop_down_menu.add_command(label="Exporter (Sauvegarder)",
                                        command=None, state=tk.DISABLED)
        self.drop_down_menu.add_separator()
        self.drop_down_menu.add_command(label="Quitter",
                                        command=self.on_closing)
        self.toolbar.add_cascade(label="Fichier",
                                 menu=self.drop_down_menu)  # permet l'affichage du menu défilant dans la barre de menu

        # création du bouton d'aide
        self.toolbar.add_command(label="Aide", command=self.aide)

        self.help_w = None  # permet de savoir si la fenêtre n'est pas déjà ouverte

    def on_closing(self):
        """méthode appellait lorsque l'utilisateur tente de fermer la fenêtre"""
        if messagebox.askokcancel("Quitter", "Quitter la simulation ?\n"):
            self.root.destroy()

    def importation(self):
        """Open the video file and play it"""
        video_file = tk.filedialog.askopenfilename(
            filetypes=(("fichiers mp4", "*.mp4"), ("all files", "*.*")))
        PlayVideo(video_file)


    # création de la fenêtre d'aide
    def aide(self):
        """ouvre une fenêtre texte contenant des informations sur l'application"""

        # 2ème condition pour si la fenêtre a été fermé et on veut la réouvrir
        if self.help_w is None or not self.help_w.winfo_exists():
            self.help_w = tk.Toplevel(self.root)
            self.help_w.title("Aide")
            title = """Bienvenue dans l'aide de mon application !\n"""
            text_simulation = """
            modèle :
            --------
            2 modèles sont pour l'instant disponible depuis cette IHM,
            YOLOv8 segmentation de Ultralytics (model seg) et le notre YOLOUni (model box).
            
            Les buttons sous les 2 boutons de modèles permettent de sélectionner les objets à flouter.
            
            Notre modèle a du mal a détecté les objets désiré, il aurait besoin d'un entraînement plus important 
            sur un data set beaucoup plus large que celui utilisé.
            
            Fichiers :
            ----------
            Dans fichier cliquer sur 'Importer' pour sélectionner une vidéo du projet."""

            text_author = "\n\nAuteur :\n-------\nThéo CARON, Adrien BRUN"

            label_title = ttk.Label(self.help_w, text=title,
                                    font=("Calibri", 12))
            text_label = ttk.Label(self.help_w,
                                   text=text_simulation + text_author)
            # scroll_bar = ttk.Scrollbar(self.help_w)

            label_title.pack()
            text_label.pack()
            # scroll_bar.pack(side='right', fill=tk.Y)


filt_box = ["real", "LicensePlate"]
filt_seg = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
#model = YOLO('yolov8n-seg.pt')  # official model
model = YOLO('./yoloUni.pt')  # custom model
TYPE_MODEL = MODEL_BOX

# --- fonctions de changements de mode ---
def model_seg():
    global model, TYPE_MODEL
    model = YOLO('yolov8n-seg.pt')  # official model
    TYPE_MODEL = MODEL_SEG

def model_uni():
    global model, TYPE_MODEL
    model = YOLO('yoloUni.pt')  # custom model
    TYPE_MODEL = MODEL_BOX


# --- fonctions de floutage ---
def blurring_box(results):
    global filt_box
    annotated_frame = results[0].plot(boxes=False, masks=False)
    k = 0
    # Making a blurred version of the frame
    blurred_frame = cv2.blur(annotated_frame, (30, 30), 0)
    mask = np.zeros((blurred_frame.shape[0], blurred_frame.shape[1], 3),
                    dtype=np.uint8)  # Making a big empty matrix
    for box in results[0].boxes:  # for each mask containing segments
        if results[0].names[int(results[0].boxes.cls[k].item())] in filt_box:
            b = box.xyxy[0]
            mask = cv2.rectangle(mask, (int(b[0]), int(b[1])),
                                 (int(b[2]), int(b[3])), (255, 255, 255), -1)
    annotated_frame = np.where(mask == (0, 0, 0), annotated_frame,
                               blurred_frame)  # Apply the mask
    return annotated_frame


def blurring_seg(results):
    global filt_seg
    # Visualize the results on the frame (not printing the boxes/masks/labels)
    annotated_frame = results[0].plot(boxes=False, masks=False)
    # annotated_frame = results[0].plot() #show mask + boxes

    # Make a mask around an object with segments and blurring it
    k = 0
    blurred_frame = cv2.blur(annotated_frame, (30, 30), 0)  # Making a blurred version of the frame
    mask = np.zeros((blurred_frame.shape[0], blurred_frame.shape[1], 3),
                    dtype=np.uint8)  # Making a big empty matrix
    if results[0].masks:
        for segs in results[0].masks.xy:  # for each mask containing segments
            if (results[0].names[int(
                    results[0].boxes.cls[k].item())] in filt_seg):  # Filtering what we don't want to blur
                segs = segs.astype(np.int32, copy=False)  # float to int
                mask = cv2.fillPoly(mask, [segs],
                    (255, 255, 255))  # Fill the poly that will become the mask
            k += 1
        annotated_frame = np.where(mask == (0, 0, 0), annotated_frame,
                                   blurred_frame)  # Apply the mask
    return annotated_frame


cap = None
annotated_frame = None

# --- fonctions de lecture et d'affichage vidéo ---
def PlayVideo(video_file):
    global cap
    cap = cv2.VideoCapture(video_file)  # mettre 0 pour avoir la cam
    while cap.isOpened():
        # Read a frame from the video
        show_frame()  # Display 2
        window.mainloop()

def show_frame():
    global cap, annotated_frame
    success, frame = cap.read()
    if success:
        results = model.predict(frame, conf=0.2)
        if TYPE_MODEL == MODEL_BOX:
            annotated_frame = blurring_box(results)
        else:
            annotated_frame = blurring_seg(results)

        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
    else:
        window.destroy()


# --- fonctions de filtrage ---
def WindowToggle():
    Toggle(toggle_window, "windows", "window-close", filt_box)

def FaceToggle():
    Toggle(toggle_face, "face", "real", filt_box)

def PlatesToggle():
    Toggle(toggle_plates, "plates", "LicensePlate", filt_box)


def PersonToggle():
    Toggle(toggle_person, "person", "person", filt_seg)

def BicycleToggle():
    Toggle(toggle_bicycle, "bicycle", "bicycle", filt_seg)

def CarToggle():
    Toggle(toggle_car, "car", "car", filt_seg)

def MotorcycleToggle():
    Toggle(toggle_motorcycle, "motorcycle", "motorcycle", filt_seg)

def AirplaneToggle():
    Toggle(toggle_airplane, "airplane", "airplane", filt_seg)

def BusToggle():
    Toggle(toggle_bus, "bus", "bus", filt_seg)


def Toggle(btn, name, toggle, filt):
    if btn.config('text')[-1] == ">" + name + "<":
        btn.config(text=name)
        filt.remove(toggle)
    else:
        btn.config(text=">" + name + "<")
        filt.append(toggle)



# Set up GUI
window = tk.Tk()  # Makes main window
window.wm_title("yoloBlurThingie")
window.config(background="#FFFFFF")
window.minsize(853, 480)
window.maxsize(1920, 1080)

button_frame = tk.Frame(window)
button_frame.pack(side="left", anchor="ne", fill=tk.Y)

button_box = tk.Button(button_frame, text="model box", width=10, command=model_uni)
button_box.pack(side="top", pady=10)

# boutons de filtrage pour model box
toggle_window = tk.Button(button_frame, text=">windows<", width=10, command=WindowToggle)
toggle_window.pack(side="top")
toggle_face = tk.Button(button_frame, text=">face<", width=10, command=FaceToggle)
toggle_face.pack(side="top")
toggle_plates = tk.Button(button_frame, text=">plates<", width=10, command=PlatesToggle)
toggle_plates.pack(side="top")

tk.Label(button_frame, height=5).pack(side="top")  # espacement

button_seg = tk.Button(button_frame, text="model segment", width=12, command=model_seg)
button_seg.pack(side="top", pady=10)

# boutons de filtrage pour model seg
toggle_person = tk.Button(button_frame, text=">person<", width=10, command=PersonToggle)
toggle_person.pack(side="top")
toggle_bicycle = tk.Button(button_frame, text=">bicycle<", width=10, command=BicycleToggle)
toggle_bicycle.pack(side="top")
toggle_car = tk.Button(button_frame, text=">car<", width=10, command=CarToggle)
toggle_car.pack(side="top")
toggle_motorcycle = tk.Button(button_frame, text=">motorcycle<", width=10, command=MotorcycleToggle)
toggle_motorcycle.pack(side="top")
toggle_airplane = tk.Button(button_frame, text=">airplane<", width=10, command=AirplaneToggle)
toggle_airplane.pack(side="top")
toggle_bus = tk.Button(button_frame, text=">bus<", width=10, command=BusToggle)
toggle_bus.pack(side="top")


# Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.pack(side="top", padx=10)

# Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)

Toolbar(window)
window.mainloop()
