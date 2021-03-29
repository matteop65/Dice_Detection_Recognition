import cv2
import numpy as np
import os

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

# Load Yolo
weights = os.path.join(trained_yolo, "yolov3.weights")
cfg = os.path.join(trained_yolo, "yolov3.cfg")
net = cv2.dnn.readNet(weights, cfg)
data_classes = os.path.join(trained_yolo, "coco.names")

# labels = []
with open(data_classes, "r") as f:
    labels = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(labels), 3))
print(labels)
# Load Images from input_images folder
folder  = "input_images"
filename = "desk1.jpg"

webcam = cv2.VideoCapture(0)


while True:
    _, img = webcam.read()
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

    # print(x)
    # print(y)
    # print(w)
    # print(labels[2])
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x, y), (x + 50*len(label), y-30), color, -1)
            cv2.putText(img, label +",{:.2f}".format(confidences[i]), (x, y-5), font, 1, (255,255,255), 2)
            print(label + ": {:.2f}".format(confidences[i]))
            # if label == "laptop":
            #     print("im here")
            #     filename2 = "crop.jpg"
            #     folder = "output_images"
            #     image = cv2.imread(os.path.join(folder, filename2))
            #     crop_img = img[y:y+h, x:x + w]
            #     cv2.imwrite(os.path.join(folder, filename2), crop_img)
            #     cv2.imshow("crop", crop_img)
            #     cv2.waitKey(0)

    folder = "output_images"

    if cv2.waitKey(1) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
    # cv2.imwrite(os.path.join(folder, filename), img)
    cv2.imshow("Image", img)
    # # cv2.waitKey(10000)
    # cv2.waitKey(0)
    #
    # crop_img = img[x:x+w]
    # cv2.imwrite(os.path.join(folder, "crop"), crop_img)
