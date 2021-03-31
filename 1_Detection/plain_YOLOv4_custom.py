import os
import cv2
import numpy as np
import time

# main direction (program)
def get_parent_dir(n=1):
    """returns the n-th parent directory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


# initialising directories
detection_folder = os.path.join(get_parent_dir(1), "1_Detection")
camera_inputs = os.path.join(detection_folder, "Camera_Inputs")
annotated_images = os.path.join(detection_folder, "Annotated_Images")
trained_yolo = os.path.join(get_parent_dir(1), "Trained_YOLO")

# load YOLO
weights = os.path.join(trained_yolo, "yolov4-custom.weights")
cfg = os.path.join(trained_yolo, "custom-yolov4-detector.cfg")
net = cv2.dnn.readNet(weights, cfg)
labels = ["Dice", "Dice Face"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialising camera
webcam = cv2.VideoCapture(0)

while True:
    time.sleep(0.5)
    _, img = webcam.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # setting up coordinates to be labelled
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h, center_x, center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # NMS Algorithm
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Setting up labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                # labelling
                x, y, w, h, center_x, center_y = boxes[i]
                label = str(labels[class_ids[i]])
                if label == "Dice":
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(img, (x, y), (x + 50 * len(label), y - 30), color, -1)
                cv2.putText(img, label + ",{:.2f}".format(confidences[i]), (x, y - 5), font, 1, (255, 255, 255), 2)
                print(label + ": {:.2f}".format(confidences[i]))
    cv2.imshow("image", img)
    cv2.waitKey(500)