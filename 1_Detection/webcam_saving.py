import os
import cv2
import time
import numpy as np

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
data = os.path.join(get_parent_dir(1), "Data")
training_images = os.path.join(data, "Training_Images")
source_images = os.path.join(data, "Source_Images")

camera_inputs = os.path.join(detection_folder, "Camera_Inputs")
annotated_images = os.path.join(data, "Annotated_Images")
temporary_images = os.path.join(detection_folder, "Temporary_Images")
trained_yolo = os.path.join(get_parent_dir(1), "Trained_YOLO")


# load YOLO
weights = os.path.join(trained_yolo, "yolov4-custom.weights")
cfg = os.path.join(trained_yolo, "custom-yolov4-detector.cfg")
net = cv2.dnn.readNet(weights, cfg)
labels = ["Dice", "Dice Face"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialising variables
colors = np.random.uniform(0, 255, size=(len(labels), 3))
iterations_for_dice_number = [0]
number_of_dice_per_iteration = 0
dice_number = 0
number = 0
previous_x = []
previous_y = []
previous_w = []
previous_h = []
previous_center_x = []
previous_center_y = []

# initialise webcam
webcam = cv2.VideoCapture(0)
done = True

# save a copy of the original image
outfile = os.path.join(temporary_images, 'dice_copy.jpg')
_, copy_img = webcam.read()
cv2.imwrite(os.path.join(temporary_images, "temporary.jpg"), copy_img)


while done:
    number_of_dice_per_iteration = 0
    # time.sleep(2)
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
            number_of_dice_per_iteration = number_of_dice_per_iteration + 1
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
            number = number + 1
            outfile = os.path.join(camera_inputs, 'dice_%s.jpg' % str(number))
            cv2.imwrite(outfile, img)

            # save a copy of the original image
            # with its coordinates. To be called if saving.
            previous_x.append(x)
            previous_y.append(y)
            previous_w.append(w)
            previous_h.append(h)
            previous_center_x.append(center_x)
            previous_center_y.append(center_y)


    number = 0
    print("previous y: " + str(previous_y))
    print("number of dice: " + str(number_of_dice_per_iteration))
    # determining whether all dice have been read
    iterations_for_dice_number.append(number_of_dice_per_iteration)
    if iterations_for_dice_number[len(iterations_for_dice_number) - 2] == number_of_dice_per_iteration:
        # then save images of separate dice
        for i in range(len(boxes)):
            if i in indexes:
                # setting up indexing and box values
                d = len(previous_x) - number_of_dice_per_iteration + number
                x = previous_x[d]
                y = previous_y[d]
                h = previous_h[d]
                w = previous_w[d]
                print("index: "+ str(d))

                # identifying labels
                label = str(labels[class_ids[i]])

                # opening temporary image
                image = cv2.imread(os.path.join(temporary_images, "temporary.jpg"))

                dice_number = dice_number + 1
                output_file = os.path.join(data, "Source_Images", "Annotated_Images", "dice_%s.jpg" % str(dice_number))
                # image = cv2.imread(os.path.join(temporary_images, "temporary.jpg"))
                crop_img = image[y:y+h, x:x+w]
                cv2.imwrite(output_file, crop_img)
                print('dice_%s.jpg' % str(dice_number))
                number = number + 1
        done = False
    else:
        continue

    # cv2.imshow("image", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     webcam.release()
    #     cv2.destroyAllWindows()
    #     break


cv2.imshow("image", img)
cv2.waitKey(0)