# Dice_Detection_Recognition
 Version 1.0

 This repo contains the code required to run a dice recognition software and also train the machine learning algorithms.

 2 YOLO (You Only Look Once) algorithms were trained based on Darknet:

    1. To detect the number of dice (mAP 91.77%)
    2. To detect the dice value (mAP 85.81% distributed between classes as: 
        One-17.95%, 
        Two-98.65%, 
        Three-100%, 
        Four-98.25%, 
        Five-100%, 
        Six-100%)

 The first algorithm is used to verify that all the dice values (in the second algorithm) have been detected. 

 The current version only works with:
    - 6 sided dice
    - pins as numerical values
    - primarily wood backgrounds

This can be easily be expanded, just needs lots of training for the verious scenarios. 
Please feel free to add any images to the training datasets (can be found in Data/Training_Images). 

## Running
before you start make sure you install the requirements. At the moment only opencv and numpy are required, but more may be added in the future. To do so, make sure you have pip installed and run the following command in a virtual environment - in the main directory: 
    > pip3 install -r requirements.txt

Point your camera to the dice (or where they will be thrown), run: 
    > python3 ./main.py
and then it should do the rest automatically. 

I have not tested whether the software works with python 2.x.

## Training Data
 In the directory (Data/Training_Images) you will find three folders, Test, Train and Valid. The Test folder is used to verify of any results if you want to test them after the algorithm is trained. I used the Train folder for the training algorithm to improve, and the Valid folder for images to calculate the mAP. I did not bother to test the results on an additional set of Images, but you can if you want, just make sure to add the images. 

 All images in the Train and Valid folders should be labelled in Darknet YOLO format, i.e. a txt that contains the class number, initial position (x and y) and width and height of the labelled bounding boxes. 

 I suggest using labelImg (Link HERE)- just make sure to delete the classes that come with the software. Other alternatives include using VoTT and then running the results through a conversion to get them into txt format. 

 To train the dataset I used Google Colab. I have provided the Notebook called (...). Make sure that you set it up with drive, that way if anything goes wrong there is a backup. For any additional information consults the README.md in the Training Folder. 


 
