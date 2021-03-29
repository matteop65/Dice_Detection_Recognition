import cv2
import os
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
camera_inputs = os.path.join(detection_folder, "Camera_Inputs")
annotated_images = os.path.join(detection_folder, "Annotated_Images")
trained_yolo = os.path.join(get_parent_dir(1), "Trained_YOLO")

# initialising camera
vid = cv2.VideoCapture(0)

# initialising variables
number = 0

while True:
    # Load YOLO
    weights = os.path.join(trained_yolo, "yolov3.weights")
    cfg = os.path.join(trained_yolo, "yolov3.cfg")
    net = cv2.dnn.readNet(weights, cfg)
    data_classes = os.path.join(trained_yolo, "data_classes.txt")
    # just for COCO with darknet do this.
    # Otherwise make use of a txt file and set the labels equal to that.
    with open(data_classes, "r") as f:
        labels = [line.strp() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0,255, size=(len(labels),3))

    # Load Images

    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    # obj_detected = 0;
    # saves image to file when you press p
    # idea being if object detected, then take picture
    if cv2.waitKey(1) & 0xFF == ord('p'):
        number = number + 1
        outfile = os.path.join(camera_inputs, 'dice_%s.jpg' % str(number))
        cv2.imwrite(outfile, frame)

    # closes file
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
