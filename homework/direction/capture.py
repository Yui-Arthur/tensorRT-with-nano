import cv2
from pathlib import Path


# create datset folder
Path('data/direction_dataset').mkdir(parents=True, exist_ok=True) 

# Capture Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("VideoCapture Error")
    exit(1)

# TODO set the inference img size #
WIDTH = 0
HEIGHT = 0
#########################

start_save = False
# start capture
print("Starting Capture")
while(True):
    ret, frame = cap.read()

    if not ret:
        break
    
    # show the frame
    cv2.imshow('frame', frame)
    

    if start_save:
        cv2.resize(frame , (WIDTH , HEIGHT))    
        # TODO save the frame in class folder or save the frame's filename with class id
        # save the frame
        cv2.imwrite('data/direction_dataset/sample.jpg', frame)
        #########################

    key = cv2.waitKey(90)
    print(key)
    # press q to quit / s to start save frame
    if  key == ord("q"):
        print("Quit")
        break
    elif key == ord("s"):
        print("Start save frame")
        start_save = True

cv2.destroyAllWindows()
