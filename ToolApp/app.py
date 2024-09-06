from __future__ import print_function  # Allows print to be used as a function in Python 2.x
import cv2  # OpenCV for image processing
from PIL import Image, ImageTk  # Image processing and Tkinter-compatible image format
import customtkinter
import tkinter as tki  # Standard Python interface to the Tk GUI toolkit
from customtkinter import *
import tkinter as tk  # Standard Python interface to the Tk GUI toolkit
from tkinter import ttk  # Themed widgets for Tk
from tkinter import PhotoImage, StringVar  # Tkinter-compatible image format and string variable
import customtkinter as ctk  # Custom Tkinter for enhanced UI elements
from functools import partial  # Facilitates the creation of partial functions
import threading  # Support for concurrent execution
import datetime  # Basic date and time types
import imutils  # Convenience functions for image processing
import os  # Miscellaneous operating system interfaces
import numpy as np  # Fundamental package for scientific computing
from requests import get  # HTTP library for sending requests
from wificonnect4 import CreateWifiConfig  # Custom module for WiFi configuration
from emailing import sendEmail  # Custom module for sending emails
from helpers import *  # Import all functions from helpers module
import time  # Time access and conversions


# Set appearance mode for customtkinter (dark mode, light mode, system default)
ctk.set_appearance_mode("dark-blue")

def validate(name, password):
    # Logs user into wifi (name) using given password
    CreateWifiConfig(name.get(), password.get())

class PhotoApp:
    def __init__(self, vs, outputPath=""):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        # Attribute to keep track of socket detection
        self.isSocket = False

        # initialize the root window and image panel
        self.root = ctk.CTk()

        #initialize images using CTkImage instead of PhotoImage
        #self.sign_in_image = ImageTk.PhotoImage(file='images/sign-in.png', master=self.root, width=100, height=100)
       
        # Load the image using PIL
        light_image_path = 'images/sign-in.png'  # Make sure this path is correct
        light_image = Image.open(light_image_path)  # This loads the image as a PIL Image object

        dark_image_path = 'images/sign-in.png'  # Make sure this path is correct
        dark_image = Image.open(dark_image_path)
        
        # Create a CTkImage object
        self.sign_in_image = ctk.CTkImage(light_image=light_image, 
                                        dark_image=dark_image, 
                                        size=(620, 620))

        image_label = ctk.CTkLabel(self.root, image=self.sign_in_image, text='')



        # Allows you to press Ctrl + f to shutdown the app, otherwise, user can't exit it
        self.root.bind('<Control-f>', self.quitP)
        # this is a hack to get the window to show up in full screen
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        self.panel = None
        self.root.attributes('-fullscreen', True)

        # Holds captured images
        self.pics = []
        # Holds non-thresholded images
        self.fullPics = []

        # number of currently saved images
        self.picCount = 0

        # label text to confirm a submission
        self.confirmLabel = False

        # Parameters and mappings specific to camera used that will undistort fisheye effect
        self.kernel_size_3_3 = (3, 3) # kernel size for morphological operations
        self.kernel_size_5_5 = (5, 5) # kernel size for morphological operations
        DIM = (1920, 1440) # Camera resolution
        K = np.array([[1901.928805723592, 0.0, 980.0566494093395], [0.0, 1900.9244832295642, 729.9864355009532], [0.0, 0.0, 1.0]]) # Camera matrix
        D = np.array([[-0.08845786239821046], [-1.296911560332019], [7.667798612427915], [-15.350865045022662]]) # Distortion coefficients
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2) # Mapping for undistortion


        # Check Internet Connection
        try:
            req = get('http://clients3.google.com/generate_204', timeout=5)
            print("Status Code:", req.status_code)  # Debug print
            if req.status_code == 204:
                print("Internet Connection Detected")  # Debug print
                self.getName()
            else:
                # Setup wifi login screen using customtkinter
                print("No Internet Connection Detected") 
                self.setup_wifi_configure()
        except Exception as e:
            print("Exception occurred:", str(e)) 

            


    def setup_wifi_config(self):
        """
        Setup the wifi configuration screen using customtkinter.
        """

        #place image
        image_label = ctk.CTkLabel(self.root, image=self.sign_in_image, text='')
        image_label.place(relx=0.2, rely=0.5, anchor='center')
        

        # Use CTkLabel for the title
        title = ctk.CTkLabel(self.root, text="Login to your Wifi to Begin", font=('yu gothic ui', 25, 'bold'), text_color='#abe620', bg_color='black')
        title.place(relx=0.5, rely=0.1, anchor='center')

        
        #Wifi Name Entry
        name_label = ctk.CTkLabel(self.root, text="Wifi Name", font=('yu gothic ui', 13, 'bold'), text_color='#abe620')
        name_label.place(relx=0.7, rely=0.3, anchor='center')
        name_entry = ctk.CTkEntry(self.root, placeholder_text="SSID", width=300, text_color='#abe620')
        name_entry.place(relx=0.7, rely=0.35, anchor='center')

        #Password Entry
        password_label = ctk.CTkLabel(self.root, text="Password", font=("yu gothic ui", 13, "bold"), text_color='#abe620')
        password_label.place(relx=0.7, rely=0.4, anchor='center')
        password_entry = ctk.CTkEntry(self.root, placeholder_text="Password", width=300, text_color='#abe620', show="*")
        password_entry.place(relx=0.7, rely=0.45, anchor='center')

        #Login Button
        login_button = ctk.CTkButton(self.root, text="Login", corner_radius=32, fg_color='transparent',
                                    hover_color='#abe620', border_color='#abe620', border_width=2, 
                                    command=partial(validate, name_entry, password_entry))
        login_button.place(relx=0.7, rely=0.6, anchor='center')


       
    def getName(self):
        """
        Get the name of the customer along with other desired information.

        Returns:
            None
        """

        self.image_label = ctk.CTkLabel(self.root, image=self.sign_in_image, text='')
        self.image_label.place(relx=0.2, rely=0.5, anchor='center')

        
        info_frame = ctk.CTkFrame(self.root, corner_radius=10, fg_color="#6b6a69")  # Set fg_color to "white" for the frame background
        info_frame.place(relx=0.5, rely=0.2, relwidth=0.4, relheight=0.6)  # Adjust placement and size as needed

        # Now, add your customer information fields inside this frame instead of directly to self.root
        # Adjust the relative x and y placements as they are now relative to info_frame
        title = ctk.CTkLabel(info_frame, text="Client Information", font=("yu gothic ui", 25, "bold"), text_color='#abe620')
        title.pack(pady=(10, 20))  # Adjust padding as needed

        name_label = ctk.CTkLabel(info_frame, text="Customer/ Organization Name", font=("yu gothic ui", 13, "bold"), text_color='#abe620')
        name_label.pack(pady=(0, 10))
        self.cname = ctk.CTkEntry(info_frame, placeholder_text="Name", width=300, text_color='#abe620')  # Store as attribute
        self.cname.pack(pady=(0, 20))

        email_label = ctk.CTkLabel(info_frame, text="Email", font=("yu gothic ui", 13, "bold"), text_color='#abe620')
        email_label.pack(pady=(0, 10))
        self.email = ctk.CTkEntry(info_frame, placeholder_text="E-mail", width=300, text_color='#abe620')  # Store as attribute
        self.email.pack(pady=(0, 20))

        phone_label = ctk.CTkLabel(info_frame, text="Phone Number", font=("yu gothic ui", 13, "bold"), text_color='#abe620')
        phone_label.pack(pady=(0, 10))
        self.phone = ctk.CTkEntry(info_frame, placeholder_text="Phone Number", width=300, text_color='#abe620')  # Store as attribute
        self.phone.pack(pady=(0, 20))

        # Start Button
        start_button = ctk.CTkButton(info_frame, text="Start", corner_radius=32, fg_color='transparent',
                                hover_color='#abe620', border_color='#abe620', border_width=2,
                                command=self.preShowMenu)
        start_button.pack(pady=(20, 0))


    

    def preShowMenu(self):
        #Starts up camera display for staring a batch menu

        # removes all previous items on the screen
        for widget in self.root.winfo_children():
            widget.destroy()

        self.videoShoot()
        self.showMenu()

    def enterInfo(self):
        # Takes user info for batch session
        self.logonButton.place_forget()
        self.shutdownButton.place_forget()

        # Begin Session Button
        self.beginSButton = ctk.CTkButton(self.root, text="Begin Session", command=self.batchStart, corner_radius=10,
                                        fg_color='#abe620', text_color='white')
        self.beginSButton.place(relx=0.8, rely=0.4, anchor='center')

        # Label for entering a name for this batch of images
        self.batchLabel = ctk.CTkLabel(self.root, text="Enter a Name for this Batch of Images",
                                    font=("yu gothic ui", 13, "bold"), text_color='#abe620')
        self.batchLabel.place(relx=0.8, rely=0.3, anchor='center')

        # Entry for batch name
        self.bname = ctk.CTkEntry(self.root, placeholder_text="Batch Name", width=300, text_color='#abe620')
        self.bname.place(relx=0.8, rely=0.35, anchor='center')

        # Optionally clear any existing label text
        if self.confirmLabel:
            self.countLabel.configure(text='')

    def toggleSocketDetection(self):
        """
        Toggles the socket detection checkbox
        """
        # This method will be bound to the socket button
        self.isSocket = self.socketCheckVar.get()

    def batchStart(self):
        # Sets up main screen for taking snapshots

        # Remove items for specifying batch name
        self.beginSButton.place_forget()
        self.batchLabel.place_forget()
        self.bname.place_forget()

        # Button to cancel the batch
        self.btn2 = ctk.CTkButton(self.root, text="Cancel Batch", command=self.cancel, corner_radius=10, fg_color='#abe620', text_color='white')
        self.btn2.place(relx=0.8, rely=0.6, anchor='center')

        
        # create a button, that when pressed, will take the current
        # frame and save it to file
        self.tbtn = ctk.CTkButton(self.root, text="Take Snapshot", command=self.takeSnapshot, corner_radius=10, fg_color='#abe620', text_color='white')
        self.tbtn.place(relx=0.8, rely=0.4, anchor='center')

        # Button to submit the batch
        self.btn = ctk.CTkButton(self.root, text="Submit Batch", command=self.finish, corner_radius=10, fg_color='#abe620', text_color='white')
        self.btn.place(relx=0.8, rely=0.8, anchor='center')

        # Button to cancel the batch
        self.btn2 = ctk.CTkButton(self.root, text="Cancel Batch", command=self.cancel, corner_radius=10, fg_color='#abe620', text_color='white')
        self.btn2.place(relx=0.8, rely=0.6, anchor='center')

        #Socket Detection Checkbox
        self.socketCheckVar = tk.BooleanVar(value=False)
        self.socketCheckBox = tk.Checkbutton(self.root, text="Socket", variable=self.socketCheckVar, command=self.toggleSocketDetection)
        self.socketCheckBox.place(relx=0.8, rely=0.45, anchor='center')
    

        # Update the label for the number of images in the batch
        self.countLabel = ctk.CTkLabel(self.root, text="Images in Batch " + str(self.picCount), text_color='#abe620')
        self.countLabel.place(relx=.8, rely=0.03, anchor='center')

        

    def videoShoot(self):
        # Main Image capture frame

        # for widget in self.root.winfo_children():
        #    widget.destroy()
        # self.beginSButton.place_forget()
        # self.batchLabel.place_forget()
        # self.batchEntry.place_forget()

        # Undo button is intially not visible
        self.undoVis = False
        # create a button, that when pressed, will take the current
        # frame and save it to file
        # self.tbtn = tki.Button(self.root, text="Take Snapshot",
        #    command=self.takeSnapshot, height=7, width=64)
        # btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        # self.tbtn.grid(row=2, column=0)
        # self.btn = tki.Button(self.root, text="Finish", command=self.finish, height=7, width=38)

        # self.btn.grid(row=2, column=1)

        # if self.panelHide:
        #    self.panelHide = False
        #    self.panelself.panel.grid(row=0, column=0, rowspan=2)


        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        # Threading allows multiple process to occur at once - needed to have video going alongside others
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed

    def shutdown(self):
        # Shutsdown everything
        os.system('sudo shutdown -h now')

    def showMenu(self):
        """
        Shows the main menu for the app.
        """
        # Shows start batch menu
        self.logonButton = ctk.CTkButton(self.root, text="New Session", command=self.enterInfo, corner_radius=10, fg_color='#abe620', text_color='white')
        self.logonButton.place(relx=0.8, rely=0.4, anchor='center')
        self.shutdownButton = ctk.CTkButton(self.root, text="Shutdown", command=self.shutdown, corner_radius=10, fg_color='#abe620', text_color='white')
        self.shutdownButton.place(relx=0.8, rely=0.6, anchor='center')

    
    def videoLoop(self):
        # Keeps the video going in the app
        # keep looping over frames until we are instructed to stop
        while not self.stopEvent.is_set():
            # grab the frame from the video stream and resize it to
            # have a maximum width of 300 pixels
            _, self.frame0 = self.vs.read()

            # Resize the frame to have a maximum width of 640 pixels
            target_width = 640
            aspect_ratio = 16 / 9
            target_height = int(target_width / aspect_ratio)

            self.frame = imutils.resize(self.frame0, width=target_width, height=target_height)

            # OpenCV represents images in BGR order; however PIL
            # represents images in RGB order, so we need to swap
            # the channels, then convert to PIL and ImageTk format
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # if the panel is not None, we need to initialize it
            if self.panel is None:
                self.panel = tki.Label(image=image)
                self.panel.image = image
                # self.panel.pack(side="left", padx=10, pady=10)
                self.panel.grid(row=0, column=0, rowspan=3, columnspan=2)

            # otherwise, simply update the panel
            else:
                self.panel.configure(image=image)
                self.panel.image = image

    def takeSnapshot(self):
        # Take a snapshot from the video feed
        # grab the current timestamp and use it to construct the
        # output path

        # name the reference file to save
        #p = "{}.jpg".format(self.cname.get())


        frame = cv2.cvtColor(self.frame0.copy(), cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('start' + str(self.picCount) + '.jpg', frame.copy())

        # Correct for fisheye
        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Cutout appropriate portion of image (whiteboard portion)
        frame = frame[357:1145, 685:1320]

        # cv2.imwrite('remap' + str(self.picCount) + '.jpg', frame.copy())
        self.fullPic = frame

        # Threshold image, apply morphological operations (opening and closing), apply blur, and apply erosion
        ret5, frame2 = cv2.threshold(frame, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size_3_3)
        closing = cv2.morphologyEx(frame2, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # opening = frame2
        opening = cv2.GaussianBlur(opening, (5, 5), 0)

        #adjust erosion size based on socket detection
        iterations = 1 if self.isSocket else 3
        opening = cv2.erode(opening, kernel, iterations=iterations)

        #cv2.imwrite(p, opening.copy())
        # save fully edited tool image
        self.curOpening = opening.copy()

        # Call the popup window to display snapshot
        self.popUp()

        #The following code was removed to use popUp instead
        '''

        #print("[INFO] saved {}".format(filename))
        width = opening.shape[1]
        height = opening.shape[0]

        thumb = ImageTk.PhotoImage(Image.fromarray(opening).resize((width//4, height//4)))
        #Setup Preview Image for possible undo


        if not self.undoVis:
            self.undo = tki.Button(self.root, text="Undo", command=self.undo, height=4, width=25)
            self.confirm = tki.Button(self.root, text="Confirm", command=self.confirm, height=4, width=25)


            self.pImg = tki.Label(self.root, image=thumb)
            # need to hold a copy for the image to show
            self.copey = thumb
            self.pImg.grid(row=0, column=2)
            self.undoVis = True
        else:
            self.pImg.config(image=thumb)
            self.copey = thumb

        self.undo.grid(row=1, column=2)
        self.confirm.grid(row=2, column=2)
        self.tbtn["state"] = "disabled"
        self.btn["state"] = "disabled"
        '''

    def popUp(self):

        # Create new popUp window
        self.top = tki.Toplevel(self.root)
        # self.top.geometry("750x250")
        self.top.title("Snapshot")
        # self.root.eval(f'tk::PlaceWindow {str(self.top)} center')

        # print("[INFO] saved {}".format(filename))
        width = self.curOpening.shape[1]
        height = self.curOpening.shape[0]

        thumb = ImageTk.PhotoImage(Image.fromarray(self.curOpening).resize((int(width // 1.75), int(height // 1.75))))

        # Add confirm and deny snapshot buttons
        self.undob = tki.Button(self.top, text="Deny", command=self.undo, height=4, width=25)
        self.confirmb = tki.Button(self.top, text="Confirm", command=self.confirm, height=4, width=25)

        self.pImg = tki.Label(self.top, image=thumb)
        self.copey = thumb
        self.pImg.grid(row=1, column=1, columnspan=2)
        # self.undoVis = True

        self.undob.grid(row=2, column=1)
        self.confirmb.grid(row=2, column=2)

    def confirm(self):
        # Confirm a Snapshot should be kept

        # Add thresholded image and full image to full list
        self.pics.append(self.curOpening)
        self.fullPics.append(self.fullPic)
        # Count number of snapshots
        self.picCount += 1

        number = "Images in Batch " + str(self.picCount)
        self.countLabel.configure(text=number)

        # Reset the chechbox for the socket detection
        self.socketCheckVar.set(False)
        self.pImg.configure(image='')
        self.undob.grid_forget()
        self.confirmb.grid_forget()
        self.tbtn["state"] = "normal"
        self.btn["state"] = "normal"

        if hasattr(self, 'top') and self.top:
            self.top.destroy()

    def undo(self):
        # Undo snapshot action, don't utilize the image taken

        self.pImg.configure(image='')
        self.undob.grid_forget()
        self.confirmb.grid_forget()
        # self.undoVis = False

        self.tbtn["state"] = "normal"
        self.btn["state"] = "normal"
        # Reset the chechbox for the socket detection
        self.socketCheckVar.set(False)


        number = "Images in Batch " + str(self.picCount)
        self.countLabel.configure(text=number)
        #ensure the top window is destroyed
        if hasattr(self, 'top') and self.top:
            self.top.destroy()

    # Called on Cancel Batch Button push, deletes all info from current batch
    def cancel(self):
        self.pics = []
        self.picCount = 0
        self.fullPics = []

        self.countLabel.configure(text='')
        self.tbtn.grid_forget()
        self.btn.grid_forget()
        # self.pImg.config(image='')
        # self.undo.grid_forget()
        # self.confirm.grid_forget()
        self.btn2.grid_forget()
        # self.panel = None
        self.showMenu()

    # Called when app shuts down
    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.close()
        self.root.quit()

    # Finish a batch and send the result via email (called upon submit batch button push)
    def finish(self):

        # combine all images into one
        imgh = self.pics[0]
        if len(self.pics) == 0:
            return

        # Concat thresholded images
        final_image = concatImages(self.pics)

        # Concat color images
        final_full_image = concatImages(self.fullPics)

        cv2.imwrite('full.bmp', final_image)
        cv2.imwrite('complete.jpg', final_full_image)

        # Vectorize
        os.system("potrace full.bmp -b dxf -W 14.5 -u .100 -O 5000 -a 0.6 -o final.dxf")

        # Email results
        sendEmail('final.dxf', 'complete.jpg', self.cname.get(), self.bname.get(), self.email.get(), self.phone.get())

        # hide all session buttons
        self.tbtn.grid_forget()
        self.btn.grid_forget()
        # self.pImg.config(image='')
        # self.undo.grid_forget()
        self.btn2.grid_forget()
        self.countLabel.configure(text='')

        textt = "Image batch {" + self.bname.get() + "} of " + str(self.picCount) + " images successfully submitted"

        # This code should be looked at, may can be deleted
        if self.confirmLabel:
            self.confirmLabel.configure(text=textt)
        else:
            self.confirmLabel = self.countLabel = tki.Label(self.root, text=textt)
            self.confirmLabel.place(relx=.4, rely=0.9, anchor='center')

        # Clean out emailed images and reset counter
        self.pics = []
        self.picCount = 0

        self.showMenu()

    def quitP(self, e):

        self.root.destroy()

# Wait a few seconds to let Wifi Connect
time.sleep(3)
# Setup openCV video stream
vs = cv2.VideoCapture(0)
#Set Camera Resolution
vs.set(3, 2048)
vs.set(4, 1536)
# Start App
pba = PhotoApp(vs)
pba.root.mainloop()

# Clean up openCV video stream
vs.release()
cv2.destroyAllWindows()



