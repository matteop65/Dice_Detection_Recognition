import cv2
import numpy as np
import os
import time

# initialising directories
main_directory = os.path.dirname(os.path.realpath(__file__))
dice_value_main = os.path.join(main_directory, "2_Dice_Value")
trained_yolo = os.path.join(main_directory, "Trained_YOLO")
dice_value_weights_cfg = os.path.join(trained_yolo, "Dice_Value")
images = os.path.join(main_directory, "Data", "Source_Images", "Annotated_Images")
valid_images = os.path.join(main_directory, "Data", "Training_Images", "Valid")


def evaluate_dice_value_():
    # print("entered dice value")
    # initiate variables
    no_of_dice = 0
    dice_labels = []

    # load YOLO
    weights = os.path.join(dice_value_weights_cfg, "yolov4-custom_85.81.weights")
    cfg = os.path.join(dice_value_weights_cfg, "dice_value_yolov4.cfg")
    net = cv2.dnn.readNet(weights, cfg)
    labels = ["One", "Two", "Three", "Four", "Five", "Six"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # open camera
    webcam = cv2.VideoCapture(0)
    _,img = webcam.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        # print('hello')
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # print('hello2')
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
    # print(indexes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    results = open("results.txt", "w")
    for i in range(len(boxes)):
        if i in indexes:
            no_of_dice = no_of_dice + 1
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            dice_labels.append(label)
            results.write(label + "\n")
            color = (255,0,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y), (x + 50*len(label), y-30), color, -1)
            cv2.putText(img, label +",{:.2f}".format(confidences[i]), (x, y-5), font, 1, (255,255,255), 2)
            # print(label + ": {:.2f}".format(confidences[i]))
    results.close()
    # r_d_value(no_of_dice)
    # r_d_labels(dice_labels)
    return no_of_dice, dice_labels



