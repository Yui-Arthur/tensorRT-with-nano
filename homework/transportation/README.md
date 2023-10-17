# Homework with Transportation Detection
## dataset video
* https://www.youtube.com/watch?v=0pkM866xh5A&t 
* https://www.youtube.com/watch?v=8mQ_P7PhiQE&t
* https://www.youtube.com/watch?v=iEm36ehmvno
* https://www.youtube.com/watch?v=UNJcrx9ArgA&t
## Step1 : label images
* use [roboflow](https://roboflow.com/) auto import video & label the frame with 3 class (car , truck , bus)
* download the video and spilt it into frame by yourself & use [labelImg](https://github.com/HumanSignal/labelImg) label the frame with 3 class (car , truck , bus)
![](/data/images/hw2_label.png)
## Step2 : ```train.ipynb``` train the yolov8 model
## Step3 : export the onnx model to trt engine
## Step4 : ```inference.py``` Infernce the model on nano with video & show the FPS & class name & number of each class in frame on OLED

## Pleasce complete the **TODO** part of the code