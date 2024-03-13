import tkinter as tki
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkFont
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
from ghostfreeshadow.networks import build_aggasatt_joint
import threading
import datetime
import webbrowser
import torch
from super_gradients.training import models
import time
import uuid  # Import uuid library to generate unique IDs
import math
from numpy import random
import face_recognition
import glob
import numpy as np
import re
import os
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


'''
vgg19_path = './ghostfreeshadow/Models/imagenet-vgg-verydeep-19.mat'
pretrain_model_path = './ghostfreeshadow/Models/srdplus-pretrained/'

with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,None,None,3])
    shadow_free_image=build_aggasatt_joint(input,64,vgg19_path)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
idtd_ckpt=tf.train.get_checkpoint_state(pretrain_model_path)
saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
print('loaded '+idtd_ckpt.model_checkpoint_path)
saver_restore.restore(sess,idtd_ckpt.model_checkpoint_path)

#iminput=cv2.imread("ghostfreeshadow\Samples\SR1.jpg",-1)
#iminput=cv2.cvtColor(iminput, cv2.COLOR_BGR2RGB)

#imoutput = sess.run(shadow_free_image,feed_dict={input:np.expand_dims(iminput/255.,axis=0)})

#imoutput = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput[0],0.0),1.0))*255.0)

#cv2.imwrite('Shadowremoved.jpg', imoutput[...,::-1])
'''

#From DeepSort program
def initialize():
    model_name = 'yolo_nas_s'
    model = load_model(model_name)

    base_path = "face/"
    # test_path = "face_detect_test/"
    image_types = [".jpg", ".jpeg", ".png"]  # Supported image types

    # Load face encodings from training images
    actor_encodings, actor_names = load_face_encodings(base_path)

    deep_sort = initialize_deep_sort()

    names = cococlassNames()
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    return model, deep_sort, names, actor_encodings, actor_names


def load_model(model_name):
    # Check if CUDA (GPU) is available and set the device accordingly
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Load the specified model
    model = models.get(model_name, pretrained_weights="coco").to(device)
    return model


def initialize_deep_sort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deep_sort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        # min_confidence  parameter sets the minimum tracking confidence required for an object detection to be considered in the tracking process
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        #nms_max_overlap specifies the maximum allowed overlap between bounding boxes during non-maximum suppression (NMS)
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        #max_iou_distance parameter defines the maximum intersection-over-union (IoU) distance between object detections
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        # Max_age: If an object's tracking ID is lost (i.e., the object is no longer detected), this parameter determines how many frames the tracker should wait before assigning a new id
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        #nn_budget: It sets the budget for the nearest-neighbor search.
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True
        )

    return deep_sort

def cococlassNames():
  class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee",
                 "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard",
                 "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple",
                 "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                 "pottedplant", "bed","diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                 "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                 "scissors", "teddy bear", "hair drier", "toothbrush" ]
  return class_names

def compute_color_for_labels(label):
    """
    Function that adds fixed color depending on the class
    """
    if label == 0:  # person  #BGR
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

#Creating a helper function
def draw_boxes(img, bbox, classNames,dict,unknown_dict, actor_encodings, actor_names, identities=None, categories=None, names=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        cat = int(categories[i]) if categories is not None else 0
        if classNames[cat] != 'person':
            continue
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[0]
        y2 += offset[0]
        id = int(identities[i]) if identities is not None else 0
        # If person id not found in dictionary, then new person is birth into scene
        if not id in dict:
            cut_out = img[y1:y2, x1:x2].copy()
            # print(cut_out)
            # Do facial recognition
            label = recognize_faces_in_group_photo(cut_out, actor_encodings, actor_names)
            # Do birth time
            start_time = time.time()
            # If facial recognition is matched, then check if person has already been birth
            if label is not None:
                found = False
                # Check person dictionary for previous facial id
                values = dict.copy().values()
                for dic_name, dic_time in values:
                    # If person is found and have not been away for > 5min, then update dictionary with new id
                    if dic_name == label and int(dic_time - start_time) < 300:
                        dict[id] = [label, dic_time]
                        found = True
                # If person not found, then add to dictionary
                if not found:
                    if id in unknown_dict:
                        start_time = unknown_dict[id][1]
                    dict[id] = [label, start_time]

        else:
            label, start_time = dict[id]
        if label is None:
            label = "Unknown #" + str(id)
            if not id in unknown_dict:
                label = "Unknown #" + str(id)
                unknown_dict[id] = [label, start_time]
            else:
                start_time = unknown_dict[id][1]

        total_time_wait = int(300) - int(time.time() - start_time)
        minutes, seconds = divmod(total_time_wait, 60)
        total_time_wait = f'{int(minutes):02d}:{int(seconds):02d}'
        # cv2.imwrite("Cut_out.jpg", cut_out)
        # Create Boxes around the detected person(s)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=compute_color_for_labels(cat), thickness=2, lineType=cv2.LINE_AA)
        # label = str(names[id-1])
        (w, h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 / 2, thickness=1)
        # Create a rectangle above the detected person, add label, and confidence score
        t_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 / 2, thickness=1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img, str(label) + " " + str(total_time_wait), (x1, y1 - 2), 0, 1 / 2, [255, 255, 255], thickness=1,
                    lineType=cv2.LINE_AA)
    return img

#From Face_recognition program
def resize_image_with_padding(img, output_size):
    h, w = img.shape[:2]
    desired_w, desired_h = output_size

    scale = min(desired_w/w, desired_h/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top_pad = (desired_h - new_h) // 2
    bottom_pad = desired_h - new_h - top_pad
    left_pad = (desired_w - new_w) // 2
    right_pad = desired_w - new_w - left_pad

    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def load_face_encodings(base_path):
    actor_encodings = []
    actor_names = []

    image_paths = glob.glob(f"{base_path}*.jpg")
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model='small')

        if encodings:
            actor_encodings.append(encodings[0])
            actor_name = os.path.basename(image_path).split('.')[0]
            actor_name = re.sub(r'\d+', '', actor_name).strip()
            actor_names.append(actor_name)

    return actor_encodings, actor_names

def add_faces(image_paths, actor_encodings, actor_names):
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model='small')

        if encodings:
            actor_encodings.append(encodings[0])
            actor_name = os.path.basename(image_path).split('.')[0]
            actor_name = re.sub(r'\d+', '', actor_name).strip()
            actor_names.append(actor_name)
    return actor_encodings, actor_names

def recognize_faces_in_group_photo(test_image, actor_encodings, actor_names, output_size=(640, 480)):
    # test_image = face_recognition.load_image_file(test_image_path)
    test_image = resize_image_with_padding(test_image, output_size)  # Resize and pad the test image
    cv2.imwrite("tesing.jpeg",test_image )
    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations, model='small')

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(actor_encodings, face_encoding, tolerance=0.65)
        name = None

        face_distances = face_recognition.face_distance(actor_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = actor_names[best_match_index]
        return name

#GUI APP
class PhotoApp:
    def __init__(self, outputPath=""):
        # Initialization and main window setup
        self.root = tki.Tk()
        self.style = ttk.Style(self.root)
        self.entries = {}
        self.profiles = []
        self.captured_images = []
        self.current_image_index = -1
        self.is_recording = False
        self.camera_mode = 'idle'
        self.previous_tab = None
        self.setup_root()
        self.create_title_label()
        self.setup_admin_access_button()
        self.admin_access_granted = False
        self.create_tabs()
        self.initialize_camera()
        self.is_paused = False
        self.video_counter = 0  # Initialize video counter to start recording
        self.actor_encodings=[]
        self.actor_names=[]
        self.pause = False


    # UI setup
    def setup_root(self):
        self.root.bind('<Control-f>', self.quitP)
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)#
        self.root.attributes('-fullscreen', True)

    def create_title_label(self):
        self.titleLabel = tki.Label(self.root, text="VecLoc", font=("Arial", 48, "bold"), foreground="#C0C0C0")
        self.titleLabel.pack(side="top", fill="x", pady=(20, 0))

        self.subTitleLabel = tki.Label(self.root, text="The Scout Regiment", font=("Arial", 24), foreground="#C0C0C0")
        self.subTitleLabel.pack(side="top", fill="x", pady=(5, 20))

    def create_tabs(self):
        self.tabControl = ttk.Notebook(self.root)
        #self.tabControl.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Information tab setup
        self.info_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.info_tab, text='Information')
        self.create_information_widgets(self.info_tab)

        # Surveillance & Security tab setup
        self.surveillance_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.surveillance_tab, text='Surveillance & Security')
        self.create_surveillance_widgets(self.surveillance_tab)

        # System Administrator tab setup
        self.admin_tab = ttk.Frame(self.tabControl)
        # Note: Temporarily add the admin tab to the notebook, it will be configured later
        self.tabControl.add(self.admin_tab, text='System Administrator')

        # Now that admin_tab is defined, we can insert the Queue Management tab before it
        admin_tab_index = self.tabControl.index(self.admin_tab)
        self.queue_management_tab = ttk.Frame(self.tabControl)
        self.create_queue_management_widgets(self.queue_management_tab)
        self.tabControl.insert(admin_tab_index, self.queue_management_tab, text='Queue Management')

        # Finalize the setup for the System Administrator tab
        self.create_admin_widgets(self.admin_tab)
        self.tabControl.tab(self.admin_tab, state='disabled')

        self.tabControl.pack(expand=1, fill="both")

    def setup_admin_access_button(self):
        self.adminAccessButton = tki.Button(self.root, text="Admin Access", command=self.request_admin_access)
        self.adminAccessButton.pack(pady=10)

    def request_admin_access(self):
        # Asking for the admin password
        password_prompt = tki.simpledialog.askstring("Admin Access", "Enter the admin password:", show='*')
        if password_prompt == "admin":  # Assuming 'admin' is the correct password
            # Show an info dialog for successful access
            tki.messagebox.showinfo("Access Granted", "Welcome to the System Administrator tab.")
            # Enable the "System Administrator" tab
            self.tabControl.tab(self.admin_tab, state='normal')
            # Select the "System Administrator" tab
            self.tabControl.select(self.admin_tab)
        else:
            tki.messagebox.showwarning("Access Denied", "The password you entered is incorrect.")

    def create_information_widgets(self, parent):
        infoContainer = tki.Frame(parent)
        infoContainer.pack(side="top", fill="both", expand=True)
        infoContainer.grid_columnconfigure(0, weight=1)
        infoContainer.grid_columnconfigure(1, weight=1)
        infoContainer.grid_columnconfigure(2, weight=1)  # Added for camera preview
        infoContainer.grid_columnconfigure(3, weight=1)  # Added for image preview

        # Left Container for form entries
        leftContainer = tki.Frame(infoContainer)
        leftContainer.grid(row=0, column=0, sticky="nsew")
        self.create_personal_info_frame(leftContainer)
        self.create_professional_info_frame(leftContainer)
        self.create_address_frame(leftContainer)
        self.create_city_state_zip_frame(leftContainer)
        self.create_contact_info_frame(leftContainer)
        self.create_button_frame(leftContainer)

        # Camera Container
        cameraContainer = tki.Frame(infoContainer)
        cameraContainer.grid(row=0, column=2, sticky="nsew", padx=(20, 0))
        self.create_camera_frame(cameraContainer)

        # Preview Container
        previewContainer = tki.Frame(infoContainer)
        previewContainer.grid(row=0, column=3, sticky="nsew", padx=(20, 0))
        self.create_preview_frame(previewContainer)

        # Frame for the Treeview table
        tableFrame = tki.Frame(parent)
        tableFrame.pack(side="bottom", fill="both", expand=True, pady=50)
        self.create_table_frame(tableFrame)

    def create_table_frame(self, parent):
        heading_font = tkFont.Font(family="Arial", size=8, weight="bold")
        style = ttk.Style()
        style.configure("Treeview.Heading", font=heading_font)

        # Define columns for the Treeview
        self.infoTable = ttk.Treeview(parent, columns=('Index', 'Prefix', 'First Name', 'Middle Name', 'Last Name',
                                                       'Suffix', 'Title', 'Organization', 'Address', 'City', 'State',
                                                       'Zip Code', 'Phone', 'Email', 'Marital Status', 'Remarks'),
                                      show='headings')
        # Configure columns
        self.infoTable.column('Index', anchor="center", width=50)
        self.infoTable.column('Prefix', anchor="center", width=20)
        self.infoTable.column('Suffix', anchor="center", width=20)
        self.infoTable.column('City', anchor="center", width=50)
        self.infoTable.column('State', anchor="center", width=20)
        self.infoTable.column('Zip Code', anchor="center", width=50)
        self.infoTable.column('Marital Status', anchor="center", width=50)  #
        self.infoTable.column('Remarks', anchor="center", width=300)

        # Configure remaining columns
        for col in ['First Name', 'Middle Name', 'Last Name', 'Title', 'Organization', 'Address', 'Phone', 'Email']:
            self.infoTable.column(col, anchor="center", width=100)

        # Configure column headings to display text centered
        for col in self.infoTable['columns']:
            self.infoTable.heading(col, text=col, anchor='center')

        # Pack the Treeview into the frame
        self.infoTable.pack(side="left", fill="both", expand=True)

        # Scrollbar for the table, aligning it to the right of the Treeview
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.infoTable.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure the Treeview to use the scrollbar
        self.infoTable.configure(yscroll=scrollbar.set)
    def create_personal_info_frame(self, parent):
        personalInfoFrame = tki.Frame(parent)
        personalInfoFrame.pack(side="top", fill="x", padx=10, pady=5)

        profileInfoSubtitle = tki.Label(personalInfoFrame, text="Profile Information", font=("Arial", 14, "bold"))
        profileInfoSubtitle.grid(row=0, column=0, columnspan=10, pady=(0, 10))

        self.create_prefix_dropdown(personalInfoFrame)
        self.create_personal_info_entries(personalInfoFrame)

    def create_prefix_dropdown(self, parent):
        prefixOptions = ["Mr.", "Ms.", "Mrs.", "Dr.", "None"]
        self.prefixVar = tki.StringVar(self.root)
        self.prefixVar.set(prefixOptions[-1])
        prefixMenu = tki.OptionMenu(parent, self.prefixVar, *prefixOptions)
        tki.Label(parent, text="Prefix").grid(row=1, column=0, padx=5, pady=5)
        prefixMenu.grid(row=1, column=1, padx=5, pady=5)

    def create_personal_info_entries(self, parent):
        label_texts = ["First Name", "Middle Name", "Last Name"]
        for i, text in enumerate(label_texts, start=1):
            label = tki.Label(parent, text=text)
            label.grid(row=1, column=i * 2, padx=5, pady=5)
            entry = tki.Entry(parent)
            entry.grid(row=1, column=i * 2 + 1, padx=5, pady=5)
            self.entries[text.replace(" ", "_").lower()] = entry

        suffixOptions = ["Jr.", "Sr.", "None"]
        self.suffixVar = tki.StringVar(self.root)
        self.suffixVar.set(suffixOptions[-1])
        suffixMenu = tki.OptionMenu(parent, self.suffixVar, *suffixOptions)
        tki.Label(parent, text="Suffix").grid(row=1, column=8, padx=5, pady=5)
        suffixMenu.grid(row=1, column=9, padx=5, pady=5)

    def create_professional_info_frame(self, parent):
        professionalInfoFrame = tki.Frame(parent)
        professionalInfoFrame.pack(side="top", fill="x", padx=10, pady=5)

        tki.Label(professionalInfoFrame, text="Title").grid(row=1, column=0, padx=5, pady=5)
        self.titleEntry = tki.Entry(professionalInfoFrame)
        self.titleEntry.grid(row=1, column=1, padx=5, pady=5)

        tki.Label(professionalInfoFrame, text="Organization").grid(row=1, column=2, padx=5, pady=5)
        self.orgEntry = tki.Entry(professionalInfoFrame)
        self.orgEntry.grid(row=1, column=3, padx=5, pady=5)

    def create_address_frame(self, parent):
        addressFrame = tki.Frame(parent)
        addressFrame.pack(side="top", fill="x", padx=10, pady=5)

        tki.Label(addressFrame, text="Address").grid(row=2, column=0, padx=5, pady=5)
        self.addressEntry = tki.Entry(addressFrame)
        self.addressEntry.grid(row=2, column=1, padx=5, pady=5)
        self.entries["address"] = self.addressEntry

        self.addressTypeVar = tki.StringVar(self.root)
        self.addressTypeVar.set("Home")
        addressTypeMenu = tki.OptionMenu(addressFrame, self.addressTypeVar, "Home", "Apartment")
        addressTypeMenu.grid(row=2, column=3, padx=5, pady=5)

    def create_city_state_zip_frame(self, parent):
        cityStateZipFrame = tki.Frame(parent)
        cityStateZipFrame.pack(side="top", fill="x", padx=10, pady=5)

        tki.Label(cityStateZipFrame, text="City").grid(row=0, column=0, padx=5, pady=5)
        self.cityEntry = tki.Entry(cityStateZipFrame)
        self.cityEntry.grid(row=0, column=1, padx=5, pady=5)
        self.entries["city"] = self.cityEntry

        tki.Label(cityStateZipFrame, text="State").grid(row=0, column=2, padx=5, pady=5)
        self.stateVar = tki.StringVar(value="Select a state")
        states = [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
            "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
            "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
        ]
        self.stateCombo = ttk.Combobox(cityStateZipFrame, textvariable=self.stateVar, values=states, state='readonly')
        self.stateCombo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        self.entries["state"] = self.stateCombo

        tki.Label(cityStateZipFrame, text="Zip Code").grid(row=0, column=4, padx=5, pady=5)
        self.zipEntry = tki.Entry(cityStateZipFrame)
        self.zipEntry.grid(row=0, column=5, padx=5, pady=5)
        self.entries["zip_code"] = self.zipEntry

    def create_contact_info_frame(self, parent):
        contactInfoFrame = tki.Frame(parent)
        contactInfoFrame.pack(side="top", fill="x", padx=10, pady=5)

        tki.Label(contactInfoFrame, text="Phone").grid(row=4, column=0, padx=5, pady=5)
        self.phoneEntry = tki.Entry(contactInfoFrame)
        self.phoneEntry.grid(row=4, column=1, padx=5, pady=5)

        tki.Label(contactInfoFrame, text="Email").grid(row=4, column=2, padx=5, pady=5)
        self.emailEntry = tki.Entry(contactInfoFrame)
        self.emailEntry.grid(row=4, column=3, padx=5, pady=5)

        tki.Label(contactInfoFrame, text="Marital Status").grid(row=4, column=4, padx=5, pady=5)
        self.maritalStatusVar = tki.StringVar(self.root)
        maritalStatusOptions = ["Single", "Married", "Divorced", "Widowed"]
        self.maritalStatusVar.set(maritalStatusOptions[0])
        maritalStatusMenu = tki.OptionMenu(contactInfoFrame, self.maritalStatusVar, *maritalStatusOptions)
        maritalStatusMenu.grid(row=4, column=5, padx=5, pady=5)

        tki.Label(contactInfoFrame, text="Remarks").grid(row=5, column=0, padx=5, pady=5, sticky='N')
        self.remarksText = tki.Text(contactInfoFrame, height=4, width=50)
        self.remarksText.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky='WE')

    def create_button_frame(self, parent):
        buttonFrame = tki.Frame(parent)
        buttonFrame.pack(side="bottom", fill="x", padx=10, pady=10)

        saveButton = tki.Button(buttonFrame, text="Save", command=self.save_data,
                                font=('Helvetica', 12, 'bold'), padx=10, pady=5)
        saveButton.pack(side="left", padx=5)

        clearButton = tki.Button(buttonFrame, text="Clear", command=self.clear_form,
                                 font=('Helvetica', 12, 'bold'), padx=10, pady=5)
        clearButton.pack(side="left", padx=5)

    def create_surveillance_widgets(self, parent):
        # Main frame for surveillance widgets, filling the entire tab
        mainFrame = tki.Frame(parent)
        mainFrame.pack(side="top", fill="both", expand=True)

        # Frame for recording controls and camera feed on the left
        recordingFrame = tki.Frame(mainFrame, bg='grey')
        recordingFrame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Displaying the camera feed
        self.cameraFeedLabel = tki.Label(recordingFrame)
        self.cameraFeedLabel.pack(fill="both", expand=True)

        # Button to toggle recording on and off, placed in the recording frame
        self.recordButton = tki.Button(recordingFrame, text="Start Recording", command=self.toggle_recording)
        self.recordButton.pack(pady=20)

        # Label to show current recording status, placed in the recording frame
        self.recordingLabel = tki.Label(recordingFrame, text="Not Recording", fg="red")
        self.recordingLabel.pack(pady=10)

        # Frame for the video playback and list on the right
        playbackFrame = tki.Frame(mainFrame, bg='lightgray')
        playbackFrame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Video list frame within the playback frame
        videoListFrame = tki.Frame(playbackFrame)
        videoListFrame.pack(side="top", fill="both", expand=True)

        # Listbox for video files
        self.videoList = tki.Listbox(videoListFrame, height=10, width=50)
        self.videoList.pack(side="left", fill="both", expand=True)

        # Scrollbar for the Listbox
        scrollbar = tki.Scrollbar(videoListFrame, command=self.videoList.yview)
        scrollbar.pack(side="right", fill="y")
        self.videoList.configure(yscrollcommand=scrollbar.set)

        # Horizontal Scrollbar for the Listbox
        hscroll = tki.Scrollbar(videoListFrame, orient="horizontal", command=self.videoList.xview)
        hscroll.pack(side="bottom", fill="x")
        self.videoList.configure(xscrollcommand=hscroll.set)

        # Button to refresh the video list
        refreshButton = tki.Button(videoListFrame, text="Refresh List", command=self.populate_video_list)
        refreshButton.pack(side="bottom", pady=5)

        # Label for displaying the video in the playback frame
        self.videoLabel = tki.Label(playbackFrame)
        self.videoLabel.pack(side="top", fill="both", expand=True, padx= (5,10), pady= 10)

        # Playback controls below the video label in the playback frame
        controlsFrame = tki.Frame(playbackFrame)
        controlsFrame.pack(side="bottom", fill="x")

        self.rewindButton = tki.Button(controlsFrame, text="<< Rewind", command=self.rewind_video)
        self.rewindButton.pack(side="left", padx=5, pady=5)

        self.playButton = tki.Button(controlsFrame, text="Play", command=self.play_selected_video)
        self.playButton.pack(side="left", padx=5, pady=5)

        self.pauseButton = tki.Button(controlsFrame, text="Pause", command=self.toggle_pause)
        self.pauseButton.pack(side="left", padx=5, pady=5)

        self.forwardButton = tki.Button(controlsFrame, text="Forward >>", command=self.forward_video)
        self.forwardButton.pack(side="left", padx=5, pady=5)

        # Initially populate the video list
        self.populate_video_list()

    def populate_video_list(self):
        self.videoList.delete(0, tki.END)  # Clear the current list

        recordings_dir = "C:\\Users\\EETTU\\PycharmProjects\\Sytem2\\recordings"
        for video_file in os.listdir(recordings_dir):
            if video_file.endswith(".mp4"):  # Assuming your videos are in .mp4 format
                self.videoList.insert(tki.END, video_file)

    def play_selected_video(self):
        selected_index = self.videoList.curselection()
        if not selected_index:
            messagebox.showwarning("Playback Error", "Please select a video to play.")
            return

        video_file = self.videoList.get(selected_index)
        video_path = os.path.join("C:\\Users\\EETTU\\PycharmProjects\\Sytem2\\recordings", video_file)

        if self.is_recording:
            messagebox.showwarning("Playback Error", "Stop recording before playing a video.")
            return

        # Start video playback in a separate thread
        self.playback_thread = threading.Thread(target=self.video_playback, args=(video_path,))
        self.playback_thread.start()

    def video_playback(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.current_frame = 0  # Keep track of the current frame
        while cap.isOpened() and not self.is_recording:
            if not self.is_paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = cap.read()
                if ret:
                    # Convert the frame to a format suitable for Tkinter
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.videoLabel.imgtk = imgtk
                    self.videoLabel.configure(image=imgtk)
                    time.sleep(0.05)  # Adjust playback speed if necessary
                    self.current_frame += 1  # Move to the next frame
                else:
                    break
            else:
                time.sleep(0.1)  # Checking the pause state again
        cap.release()

    def rewind_video(self):
        if self.current_frame > 30:  # 30 frames per second
            self.current_frame -= 30  # Move 1 second back
        else:
            self.current_frame = 0

    def forward_video(self):
        self.current_frame += 30  # Move 1 second forward

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pauseButton.config(text="Resume")
        else:
            self.pauseButton.config(text="Pause")

    def create_queue_management_widgets(self, parent):
        # Add widgets to the Queue Management tab
        label = ttk.Label(parent, text="Projected Resolution Timeframe ")
        label.pack(padx=10, pady=10)


    def create_admin_widgets(self, parent):
        self.adminFrame = tki.Frame(parent)
        self.adminFrame.pack(fill="both", expand=True)
        self.create_admin_table_frame(self.adminFrame)
        self.setup_admin_controls(self.adminFrame)

        # Container frame for admin buttons
        buttonFrame = tki.Frame(self.adminFrame)
        buttonFrame.pack(side="bottom", fill="x", pady=10)

        webBrowserButton = tki.Button(buttonFrame, text="Web Browser", command=self.open_web_browser,
                                      font=('Helvetica', 12, 'bold'))
        webBrowserButton.pack(side="top", fill="x", padx=10, pady=(0, 5))

        logoutButton = tki.Button(parent, text="Log out", command=self.logout_admin,
                                  font=('Helvetica', 12, 'bold'))
        logoutButton.pack(side="top", fill="x", padx=10, pady=(5, 0))

        shutdownButton = tki.Button(buttonFrame, text="System Shutdown", command=self.shutdown_system,
                                    font=('Helvetica', 12, 'bold'))
        shutdownButton.pack(side="top", fill="x", padx=10, pady=(5, 0))

    def shutdown_system(self):
        if messagebox.askokcancel("Shutdown", "Do you want to shut down the system?"):
            self.root.destroy()

    def logout_admin(self):
        # Reset the admin access flag
        self.admin_access_granted = False

        # Disable the System Administrator tab
        self.tabControl.tab(self.admin_tab, state='disabled')

        # Switch back to the Information tab
        self.tabControl.select(self.info_tab)

        tki.messagebox.showinfo("Logged Out", "You have been successfully logged out.")

    def open_web_browser(self):
        webbrowser.open('http://www.google.com', new=2)

    def create_admin_table_frame(self, parent):
        heading_font = tkFont.Font(family="Arial", size=8, weight="bold")
        style = ttk.Style()
        style.configure("Treeview.Heading", font=heading_font)

        # Define columns for the Treeview
        self.adminTable = ttk.Treeview(parent, columns=('Index', 'Prefix', 'First Name', 'Middle Name', 'Last Name',
                                                        'Suffix', 'Title', 'Organization', 'Address', 'City', 'State',
                                                        'Zip Code', 'Phone', 'Email', 'Marital Status', 'Remarks'),
                                       show='headings')

        # Configure column
        self.adminTable.column('Index', anchor="center", width=50)

        # Configure and shrink columns "Prefix", "Suffix", and "State"
        self.adminTable.column('Prefix', anchor="center", width=30)
        self.adminTable.column('Suffix', anchor="center", width=30)
        self.adminTable.column('City', anchor="center", width=50)
        self.adminTable.column('State', anchor="center", width=20)
        self.adminTable.column('Zip Code', anchor="center", width=50)
        self.adminTable.column('Marital Status', anchor="center", width=70)
        self.adminTable.column('Remarks', anchor="center", width=250)

        # Configure remaining columns
        for col in ['First Name', 'Middle Name', 'Last Name', 'Title', 'Organization', 'Address', 'Phone', 'Email']:
            self.adminTable.column(col, anchor="center", width=100)

        # Configure column headings to display text centered
        for col in self.adminTable['columns']:
            self.adminTable.heading(col, text=col, anchor='center')

        self.adminTable.pack(side="left", fill="both", expand=True)


    def setup_admin_controls(self, parent):
        # Search functionality
        search_frame = tki.Frame(parent)
        self.search_var = tki.StringVar()
        search_entry = tki.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tki.LEFT)
        search_button = tki.Button(search_frame, text="Search", command=self.perform_search)
        search_button.pack(side=tki.LEFT)
        search_frame.pack()

        # Edit button
        edit_button = tki.Button(parent, text="Edit", command=self.edit_selected_profile)
        edit_button.pack(side=tki.LEFT, padx=10)

        # Delete button
        delete_button = tki.Button(parent, text="Delete", command=self.delete_selected_profile)
        delete_button.pack(side=tki.RIGHT, padx=10)

    def setup_search(self, parent):
        search_frame = tki.Frame(parent)
        search_frame.pack(pady=10)
        self.search_var = tki.StringVar()
        search_entry = tki.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tki.LEFT, padx=10)
        search_button = tki.Button(search_frame, text="Search", command=self.perform_search)
        search_button.pack(side=tki.LEFT)

    def perform_search(self):
        query = self.search_var.get().lower()
        for child in self.adminTable.get_children():
            # Extracting item values to search through
            item_values = [str(val).lower() for val in self.adminTable.item(child, "values")]
            # If query matches any of the values, highlight the row
            if any(query in value for value in item_values):
                self.adminTable.selection_set(child)  # Highlight matching entry
                self.adminTable.see(child)  # Scroll to the matching entry
                return  # Remove this if you want to highlight all matches
        messagebox.showinfo("Search", "No matching entry found.")

    def edit_selected_profile(self):
        selected = self.adminTable.selection()
        if selected:
            selected_item = selected[0]
            selected_index = int(self.adminTable.item(selected_item, 'values')[0])
            selected_profile = self.profiles[selected_index - 1]  # Adjust for zero-based indexing

            self.edit_window = tki.Toplevel(self.root)
            self.edit_entries = {}
            for i, (key, value) in enumerate(selected_profile.items()):
                tki.Label(self.edit_window, text=key).grid(row=i, column=0)
                entry = tki.Entry(self.edit_window)
                entry.insert(0, value)
                entry.grid(row=i, column=1)
                self.edit_entries[key] = entry

            save_button = tki.Button(self.edit_window, text="Save Changes",
                                     command=lambda: self.save_profile_changes(selected_index, selected_item))
            save_button.grid(row=len(selected_profile) + 1, column=0, columnspan=2)

    def save_profile_changes(self, profile_index, selected_item):
        updated_profile = {key: entry.get() for key, entry in self.edit_entries.items()}
        self.profiles[profile_index - 1].update(updated_profile)

        # Update admin and information tables
        self.update_tables(profile_index, updated_profile)

        self.edit_window.destroy()
        messagebox.showinfo("Success", "Profile updated successfully.")

    def update_tables(self, profile_index, updated_profile):
        # Construct the list of values to be updated in the table rows
        updated_values = [profile_index] + list(updated_profile.values())

        # Update the adminTable
        admin_item = self.adminTable.get_children()[profile_index - 1]  # Adjust for zero-based indexing
        self.adminTable.item(admin_item, values=updated_values)

        # Update the infoTable similarly
        info_item = self.infoTable.get_children()[profile_index - 1]
        self.infoTable.item(info_item, values=updated_values)

    def delete_selected_profile(self):
        selected = self.adminTable.selection()
        if selected:
            for item in selected:
                # Get the item's values
                item_values = self.adminTable.item(item, 'values')

                # Assuming the first column in your table is a unique ID/index
                unique_id = item_values[0]

                # Delete the item from the admin table
                self.adminTable.delete(item)

                # Now find and delete the corresponding item in the info table
                for info_item in self.infoTable.get_children():
                    if self.infoTable.item(info_item, 'values')[0] == unique_id:
                        self.infoTable.delete(info_item)
                        break

                # Remove the profile from the profiles list using the unique ID
                self.profiles = [profile for profile in self.profiles if str(profile.get('Index', '')) != unique_id]

            messagebox.showinfo("Delete Profile", "Selected profile(s) deleted successfully.")
        else:
            messagebox.showwarning("Delete Profile", "Please select a profile to delete.")

    # Camera and recording methods
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        #self.cap.set(3, 2048)
        #self.cap.set(4, 1536)
        self.stopEvent = threading.Event()
        self.livethread = threading.Thread(target=self.videoLoop, args=())
        self.livethread.start()
        self.track_frame = None
        self.ret, self.frame = self.cap.read()
        #Stop event Deep_Sort
        self.stopDeepSort = threading.Event()
        self.Deep_thread = threading.Thread(target=self.run_DeepSort, args=())
        self.Deep_thread.start()
        self.update_camera_feed()

    def run_DeepSort(self):
        model, deep_sort, names, self.actor_encodings, self.actor_names = initialize()
        # Once a face is recognized, it sticks with the image
        dict = {}
        unknown_dict = {}
        while True:
            xywh_bboxs = []
            confs = []
            oids = []
            outputs = []
            if self.ret and self.frame is not None:
                if self.pause:
                    self.actor_encodings, self.actor_names = add_faces(self.path_to_add, self.actor_encodings, self.actor_names)
                    self.pause = False
                    self.path_to_add=[]
                self.final_frame = self.frame.copy()
                #cv2.imwrite('next_frame.jpg',self.final_frame)
                #iminput = cv2.imread("next_frame.jpg", -1)
                #iminput = cv2.cvtColor(iminput, cv2.COLOR_BGR2RGB)
                #self.final_frame = cv2.cvtColor(self.final_frame, cv2.COLOR_BGR2RGB)
                #imoutput = sess.run(shadow_free_image, feed_dict={input: np.expand_dims(self.frame.copy() / 255., axis=0)})
                #self.final_frame = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput[0], 0.0), 1.0)) * 255.0)
                result = model.predict(self.final_frame, conf=0.5)
                bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
                confidences = result.prediction.confidence
                labels = result.prediction.labels.tolist()
                for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                    bbox = np.array(bbox_xyxy)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((confidence * 100)) / 100
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    bbox_width = abs(x1 - x2)
                    bbox_height = abs(y1 - y2)
                    xcycwh = [cx, cy, bbox_width, bbox_height]
                    xywh_bboxs.append(xcycwh)
                    confs.append(conf)
                    oids.append(int(cls))
                xywhs = torch.tensor(xywh_bboxs)
                confss = torch.tensor(confs)
                outputs=[]
                try:
                    outputs = deep_sort.update(bbox_xywh=xywhs, confidences=confss, oids=oids, ori_img=self.final_frame)
                except:
                    print("Error")
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    draw_boxes(self.final_frame, bbox_xyxy, names, dict, unknown_dict, self.actor_encodings, self.actor_names, identities, object_id)
                # output.write(self.frame)
                self.track_frame = self.final_frame.copy()
            else:
                break

    def videoLoop(self):
        while not self.stopEvent.is_set():
            # grab the frame from the video stream and resize it to
            # have a maximum width of 300 pixels
            self.ret, self.frame = self.cap.read()

    def update_camera_feed(self):
        if not self.cap.isOpened():
            return

        # ret, frame = self.cap.read()
        if self.ret:
            # Convert the frame to a format suitable for Tkinter
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the main camera display
            self.cameraLabel.imgtk = imgtk
            self.cameraLabel.configure(image=imgtk)

            # Also update the camera feed in the left frame if recording
            '''if self.is_recording:
                self.cameraFeedLabel.imgtk = imgtk
                self.cameraFeedLabel.configure(image=imgtk)'''

            if self.track_frame is not None:
                cv2image = cv2.cvtColor(self.track_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                deep_img = ImageTk.PhotoImage(image=img)
                self.cameraFeedLabel.imgtk = deep_img
                self.cameraFeedLabel.configure(image=deep_img)


            self.cameraLabel.after(10, self.update_camera_feed)

    def toggle_recording(self):
        if self.camera_mode == 'photo_capture':
            messagebox.showerror("Operation not allowed", "Finish capturing photos before starting to record.")
            return

        if self.is_recording:
            self.stop_recording()
            self.camera_mode = 'idle'
        else:
            self.camera_mode = 'recording'
            self.start_recording()

    def start_recording(self):
        self.video_counter += 1
        self.is_recording = True
        self.recordButton.config(text="Stop Recording")
        self.recordingLabel.config(text="Recording...", fg="green")

        # Ensure the directory exists
        recordings_dir = "C:\\Users\\EETTU\\PycharmProjects\\Sytem2\\recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)

        # Format the current date and time to use in the filename
        current_time = datetime.datetime.now().strftime("%b %d %Y @ %I %M %S %p")
        output_file = os.path.join(recordings_dir, f"recording_{current_time}.mp4")

        # Initialize VideoWriter with the specified output file path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' codec
        self.out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

        # Start recording in a separate thread to avoid blocking the GUI
        self.record_thread = threading.Thread(target=self.record)
        self.record_thread.start()

    def record(self):
        frame_interval = 1.0 / 20.0  # for 20 fps
        while self.is_recording:
            start_time = time.time()
            if self.track_frame is not None:
                try:
                    self.out.write(self.track_frame)
                except Exception as e:
                    print(f"Error writing frame: {e}")
                    break  # or continue, depending on desired behavior
            # Wait for the next frame time
            while (time.time() - start_time) < frame_interval:
                time.sleep(0.001)

    def stop_recording(self):
        self.is_recording = False
        self.recordButton.config(text="Start Recording")
        self.recordingLabel.config(text="Not Recording", fg="red")
        self.out.release()
        self.record_thread.join()

    def onClose(self):
        if self.is_recording:
            self.stop_recording()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def create_camera_frame(self, parent):
        cameraFrame = tki.Frame(parent)
        cameraFrame.pack(side="top", fill="both", expand=True)

        self.cameraLabel = tki.Label(cameraFrame)
        self.cameraLabel.pack(side="top", fill="both", expand=True)

        captureButton = tki.Button(cameraFrame, text="Capture", command=self.capture_photo,
                                   font=('Helvetica', 12, 'bold'), padx=10, pady=5)
        captureButton.pack(side="bottom", pady=10)

    def create_preview_frame(self, parent):
        # Frame for the preview area
        self.previewFrame = tki.Frame(parent, width=200, height=200)  # Adjust size as needed
        self.previewFrame.pack(side="top", fill="both", expand=True)

        # Title for the preview area
        previewTitle = tki.Label(self.previewFrame, text="Preview", font=("Arial", 16, "bold"))
        previewTitle.pack(side="top", pady=(10, 0))

        # Label for displaying the image
        self.previewLabel = tki.Label(self.previewFrame)
        self.previewLabel.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Navigation frame for the Previous and Next buttons
        navFrame = tki.Frame(parent)
        navFrame.pack(side="bottom", fill="x")

        prevButton = tki.Button(navFrame, text="Previous", command=self.show_previous_image)
        prevButton.pack(side="left", padx=10, pady=10)

        deleteButton = tki.Button(navFrame, text="Delete", command=self.delete_captured_photo)
        deleteButton.pack(side="left", padx=(30, 10), pady=10)

        nextButton = tki.Button(navFrame, text="Next", command=self.show_next_image)
        nextButton.pack(side="right", padx=10, pady=10)

    # Image capture and preview methods
    def capture_photo(self):
        if self.camera_mode == 'recording':
            messagebox.showerror("Operation not allowed", "Stop recording before capturing a photo.")
            return

        first_name = self.entries["first_name"].get().strip()
        last_name = self.entries["last_name"].get().strip()

        if not first_name or not last_name:
            messagebox.showerror("Missing Information", "Please enter both First Name and Last Name.")
            return

        # Use the specified directory for saving captured images
        save_dir = "C:\\Users\\EETTU\\PycharmProjects\\Sytem2\\face"
        os.makedirs(save_dir, exist_ok=True)
        self.path_to_add= []
        for i in range(1, 6):  # Assuming you want to capture and save 5 images
            ret, frame = self.cap.read()
            if ret:
                file_name = f"{first_name} {last_name} {i}.jpg"  # Modified for a more file-friendly format
                save_path = os.path.join(save_dir, file_name)
                cv2.imwrite(save_path, frame)
                #self.actor_encodings, self.actor_names = add_face(save_path, self.actor_encodings, self.actor_names)
                self.captured_images.append(save_path)  # Append the path of the saved image
                self.path_to_add.append(save_path)
                messagebox.showinfo("Capture", f"Captured {i}/5")
                time.sleep(1)  # Pause for a second between captures
            else:
                messagebox.showerror("Capture Error", "Failed to capture image.")
                break
        self.pause = True
        self.captured_images.append(save_path)  # Store path for future reference
        self.current_image_index = len(self.captured_images) - 1  # Update current image index to the latest photo
        self.update_preview_image(save_path)  # Update the preview with the latest captured image
        self.camera_mode = 'idle'

    def delete_captured_photo(self):
        if self.current_image_index >= 0 and self.captured_images:
            # Confirm deletion
            if messagebox.askyesno("Delete Image", "Are you sure you want to delete this image?"):
                image_path = self.captured_images[self.current_image_index]
                try:
                    os.remove(image_path)  # Delete the image file
                    del self.captured_images[self.current_image_index]  # Remove from the list

                    if self.captured_images:  # If there are still images left
                        # Adjust the index if it's now out of range
                        self.current_image_index %= len(self.captured_images)
                        self.update_preview_image(self.captured_images[self.current_image_index])
                    else:  # If no images are left
                        self.current_image_index = -1
                        self.update_preview_image()  # Clear the preview

                    messagebox.showinfo("Success", "Image deleted successfully.")
                except OSError as e:
                    messagebox.showerror("Error", f"Could not delete the image: {e}")
        else:
            messagebox.showinfo("Delete Image", "No image selected for deletion.")

    def update_preview_image(self, image_path=None):
        if image_path:
            image = Image.open(image_path)
            image = image.resize((200, 200), Image.Resampling.LANCZOS)  # Adjust size as needed
            photo = ImageTk.PhotoImage(image)
            self.previewLabel.configure(image=photo)
            self.previewLabel.image = photo
        else:
            self.previewLabel.configure(image='')
            self.previewLabel.image = None

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_preview_image(self.captured_images[self.current_image_index])

    def show_next_image(self):
        if self.current_image_index < len(self.captured_images) - 1:
            self.current_image_index += 1
            self.update_preview_image(self.captured_images[self.current_image_index])

    def save_data(self):
        # Extracting all the required fields' values
        prefix = self.prefixVar.get()
        first_name = self.entries["first_name"].get().strip()
        middle_name = self.entries.get("middle_name", "").get().strip()
        last_name = self.entries["last_name"].get().strip()
        suffix = self.suffixVar.get()
        title = getattr(self, "titleEntry", tki.Entry()).get().strip()
        organization = getattr(self, "orgEntry", tki.Entry()).get().strip()
        address = self.addressEntry.get().strip()
        city = self.cityEntry.get().strip()
        state = self.stateVar.get()
        zip_code = self.zipEntry.get().strip()
        phone = self.phoneEntry.get().strip()
        email = self.emailEntry.get().strip()
        marital_status = self.maritalStatusVar.get()
        remarks = self.remarksText.get('1.0', tki.END).strip()

        # Validate required fields
        if not all([first_name, last_name, address, city, state, zip_code]):
            messagebox.showerror("Missing Information", "Please fill in all the required fields.")
            return

        # Create a dictionary for the current profile and append it to the profiles list
        current_profile = {
            "Prefix": prefix,
            "First Name": first_name,
            "Middle Name": middle_name,
            "Last Name": last_name,
            "Suffix": suffix,
            "Title": title,
            "Organization": organization,
            "Address": address,
            "City": city,
            "State": state,
            "Zip Code": zip_code,
            "Phone": phone,
            "Email": email,
            "Marital Status": marital_status,
            "Remarks": remarks
        }
        self.profiles.append(current_profile)

        # Update both the information and admin tables with the new entry
        for table in [self.infoTable, self.adminTable]:
            table.insert('', 'end', values=(
                len(self.profiles),  # Index
                prefix,
                first_name,
                middle_name,
                last_name,
                suffix,
                title,
                organization,
                address,
                city,
                state,
                zip_code,
                phone,
                email,
                marital_status,
                remarks
            ))

        # Notify the user that the data has been saved
        messagebox.showinfo("Success", "Profile saved successfully.")

        # Clear the form for new entries
        self.clear_form()

    def clear_form(self):
        # Reset all entry widgets to empty strings
        for entry in self.entries.values():
            entry.delete(0, tki.END)

        # Reset the specific entry widgets not in the entries dictionary
        if hasattr(self, "titleEntry"):
            self.titleEntry.delete(0, tki.END)
        if hasattr(self, "orgEntry"):
            self.orgEntry.delete(0, tki.END)
        if hasattr(self, "addressEntry"):
            self.addressEntry.delete(0, tki.END)
        if hasattr(self, "cityEntry"):
            self.cityEntry.delete(0, tki.END)
        if hasattr(self, "zipEntry"):
            self.zipEntry.delete(0, tki.END)
        if hasattr(self, "phoneEntry"):
            self.phoneEntry.delete(0, tki.END)
        if hasattr(self, "emailEntry"):
            self.emailEntry.delete(0, tki.END)

        # Reset the dropdowns to their default values
        self.prefixVar.set("None")
        self.suffixVar.set("None")
        self.addressTypeVar.set("Home")
        self.maritalStatusVar.set("Single")
        self.stateVar.set("Select State")

        # Clear the remarks text field
        if hasattr(self, "remarksText"):
            self.remarksText.delete("1.0", tki.END)

        # Set the focus back to the First Name entry field
        if "first_name" in self.entries:
            self.entries["first_name"].focus_set()

    def refresh_admin_table(self):
        # Clear the existing entries in the admin table
        for item in self.adminTable.get_children():
            self.adminTable.delete(item)

        # Repopulate the table with updated profiles from self.profiles
        for index, profile in enumerate(self.profiles, start=1):
            self.adminTable.insert('', 'end', values=(
                index,  # Adjust the index if necessary
                profile.get("Prefix", ""),
                profile.get("First Name", ""),
                profile.get("Middle Name", ""),
                profile.get("Last Name", ""),
                profile.get("Suffix", ""),
                profile.get("Title", ""),
                profile.get("Organization", ""),
                profile.get("Address", ""),
                profile.get("City", ""),
                profile.get("State", ""),
                profile.get("Zip Code", ""),
                profile.get("Phone", ""),
                profile.get("Email", ""),
                profile.get("Marital Status", ""),
                profile.get("Remarks", "")
            ))
    def quitP(self, event=None):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def onClose(self):
        self.quitP()

if __name__ == "__main__":
    app = PhotoApp()
    app.root.mainloop()