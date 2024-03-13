# Object Detection and Tracking, Facial Recognition, and Shadow Removal - Wait Room Tracker

I built this python project largely out of personal interest but also with the hope of integrating it into the Deep Learning Course I built. This would teach students to utilize prebuilt deep learning models for novel applications. Package complications have prevented me from fully integrating it as a project.

All components are integrated into the single guishadow file, however associated packages and models have not been uploaded. 

These deep learning models are integrated into a tkinter app which all in one tracks users, their wait time (counting down from 5 minutes after entering the scene), identifies users with facial recognition, and applies shadow removal to the camera images used in tracking. The app also allows new person faces to be added to the facial recognition aspect. All these components interact to produce the final waitroom person tracker. 

## Component Details

### Object detection and tracking

The DeepSort and Yolo-Nas pre-built models were utilized for multiple object tracking and detection, respectively. DeepSort utilizes motion (Kalman Filtering) and appearance of objects to track them through the space of the image and through time. Yolo-Nas was set up to track people specifically and is integrated into the DeepSort algorithm as the person detector. Yolo-nas is available under the open source SuperGradients library.

### Facial Recognition

The face_recognition library is utilized for face recognition. This is a library in python that utilizes a pre-trained deep learning (CNN) model to encode face images which can then be compared to other faces for similarity. 

https://pypi.org/project/face-recognition/

