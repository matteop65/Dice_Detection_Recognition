import cv2
import numpy as np
import os
# from sys import argv
import time


# main direction (program)
def get_parent_dir(n=1):
    """returns the n-th parent directory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


def clear_coordinates():
    coordinates_files = open("coordinates.txt", "w")
    coordinates_files.write("")
    coordinates_files.close()


def output_coordinates(x, y, centre_x, centre_y):
    coordinate_file = open("coordinates.txt", "a+")
    coordinate_file.writelines([str(x),"\t", str(y),"\t", str(centre_x), "\t",str(centre_y), "\n"])
    coordinate_file.close()


clear_coordinates()

number = 0
# initialising directories
detection_folder = os.path.join(get_parent_dir(1), "1_Detection")
camera_inputs = os.path.join(detection_folder, "Camera_Inputs")
annotated_images = os.path.join(detection_folder, "Annotated_Images")
trained_yolo = os.path.join(get_parent_dir(1), "Trained_YOLO")
# annotated_images = os.path.join(detection_folder, "Annotated_Images")

# Load Yolo
weights = os.path.join(trained_yolo, "yolov4-custom.weights")
cfg = os.path.join(trained_yolo, "custom-yolov4-detector.cfg")
net = cv2.dnn.readNet(weights, cfg)
# data_classes = os.path.join(trained_yolo, "coco.names")

# labels = []
# with open(data_classes, "r") as f:
#     labels = [line.strip() for line in f.readlines()]
labels = ["dice", "dice face"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(labels), 3))
# colors = np.random.uniform(0, 255, size = (36, 3))

print(labels)
# Load Images from input_images folder
# folder  = "input_images"
# filename = "dice_2.jpg"
# img = cv2.imread(filename)
# img = cv2.resize(img, None, fx=0.6, fy=0.6)
# height, width, channels = img.shape

# This is for webcam
webcam = cv2.VideoCapture(0)


while True:
    # This is for webcam
    time.sleep(0.5)
    net, img = webcam.read()
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    centers_x = []
    centers_y = []
    xs = []
    ys = []
    dice_iteration = 0
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
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # output_coordinates(x,y,center_x,center_y)
                # coordinates = [ x, y, center_x, center_y]
                centers_x.append(center_x)
                centers_y.append(center_y)
                xs.append(x)
                ys.append(y)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            # label = labels
            # color = (0,0,0)
            if label == "dice":
                color = (255,0,0)
                output_coordinates(xs[i], ys[i], centers_x[i], centers_y[i])
            else:
                color = (0,255,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y), (x + 50*len(label), y-30), color, -1)
            cv2.putText(img, label +",{:.2f}".format(confidences[i]), (x, y-5), font, 1, (255,255,255), 1)
            print(label + ": {:.2f}".format(confidences[i]))
            number = number + 1
            outfile = os.path.join(camera_inputs, 'dice_%s.jpg' % str(number))
            cv2.imwrite(outfile, img)
            if label == "dice":
                dice_iteration = dice_iteration + 1
                print("im here")
                output_file = os.path.join(annotated_images, "dice_%s.jpg" % str(dice_iteration))
                image = cv2.imread("dice_2.jpg")
                crop_img = image[y:y+h, x:x + w]
                cv2.imwrite(output_file, crop_img)
                # cv2.imshow("crop", crop_img)



    # clear_coordinates()
    # folder = "output_images"
    # if cv2.waitKey(1) & 0xFF == ord('p'):

    # for webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    # for images
    # if cv2.waitKey(1) & 0xFF == ord('l'):
        # img.release()
        # cv2.destroyAllWindows()