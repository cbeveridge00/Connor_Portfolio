# Object Detection and Tracking, Facial Recognition, and Shadow Removal - Wait Room Tracker

I built this python project largely out of personal interest but also with the hope of integrating it into the Deep Learning Course I built. This would teach students to utilize prebuilt deep learning models for novel applications. Package complications have prevented me from fully integrating it as a project.

All components are integrated into the single guishadow file, however associated packages and models have not been uploaded. 

These deep learning models are integrated into a tkinter app which all in one tracks users, their wait time (counting down from 5 minutes after entering the scene), identifies users with facial recognition, and applies shadow removal to the camera images used in tracking. The app also allows new person faces to be added to the facial recognition aspect. All these components interact to produce the final wait room person tracker. 

Without a GPU available, the app is very slow due to the computational costs of running multiple models.

## Component Details

### Object detection and tracking

The DeepSort and Yolo-Nas pre-built models were utilized for multiple object tracking and detection, respectively. DeepSort utilizes motion (Kalman Filtering) and appearance of objects to track them through the space of the image and through time. Yolo-Nas was set up to track people specifically and is integrated into the DeepSort algorithm as the person detector. Yolo-nas is available under the open source SuperGradients library.

### Facial Recognition

The face_recognition library is utilized for face recognition. This is a library in python that utilizes a pre-trained deep learning (CNN) model to encode face images which can then be compared to other faces for similarity. If a face is similar enough to one in the dataset, it is considered recognized.

https://pypi.org/project/face-recognition/

### Shadow Removal

Powerful shadow removal is implemented to improve facial recognition. A pretrained VGG-19 model is utilized in the app to do the removal which is from the Ghost Free Shadow Removal Project:

https://github.com/vinthony/ghost-free-shadow-removal

### Other Details

The GUI has multiple tabs. The GUI was expanded for looks but the "Information" and "Surveillance & Security" hold the interesting parts of this project. In the 'Information' tab, the user can take pictures of a new user tied to their name. These images are added to the total face encodings for future recognition.

In the "Surveillance & Security" tab, the main part of the project can be seen running. Images from a camera are fed first to the shadow removal model, and then to the yolo-nas model for general person detection. These detections and images are then passed to DeepSort which assigns an ID to a new person and tracks older ones through time. If a person leaves and comes back to the screen, the model recognizes this and their ID remains the same. Facial recognition is applied after tracking to label each person with their name and a wait time is calculated based on time being tracked. 



