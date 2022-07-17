# COMPUTER VISION AND IOT
#THE SPARKS FOUNDATION

#TASK 1 : Object Detection

#NAME : Suhaila Ahmed Farouk



# importing computer Vision Library
import cv2
# importing Graph Plotting Library
import matplotlib.pyplot as plt

# Uploading The Image
img = cv2.imread("C:\\Users\\moham\\Desktop\\detection\\proj_part1\\local.jpeg")

# Converting BGR Image TO RGB Image for Plotting
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)


# List of  Classes
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Load The Pretrained neural Network
configPath = 'C:\\Users\\moham\\Desktop\\detection\\proj_part1\\MobileNetSSD_deploy.caffemodel'
weightPath = 'C:\\Users\\moham\\Desktop\\detection\\proj_part1\\MobileNetSSD_deploy.prototxt.txt'
net = cv2.dnn_DetectionModel('C:\\Users\\moham\\Desktop\\detection\\proj_part1\\MobileNetSSD_deploy.caffemodel' ,
                             'C:\\Users\\moham\\Desktop\\detection\\proj_part1\\MobileNetSSD_deploy.prototxt.txt')

# Default Parametrs  of The Neural Network
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Sending The Image to The Model With Confidence Set tO 30%
classids, confs, bbox = net.detect(img, confThreshold=0.30)
print(classids, bbox)


# Creating A Rectangular box and Putting The Respective Text
for ids , confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img,box ,color=(0, 255, 0), thickness=1)
    cv2.putText(img,  text=classes[ids].upper(),org=(box[0]+10,box[1]+30) ,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.6, color=(0,0,255),thickness=1)


# Displaying The Output Image
cv2.imshow("output", img)
cv2.waitKey(0)
