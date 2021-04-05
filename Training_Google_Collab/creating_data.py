# This code is useful to taking pictures with a camera.
# you can point the camera at the image you want to take and then press p to take the picture
# the picture is saved in any file you want, but you have to specify it
# If you want to train the algorithm, then save in train and valid directories
# make sure you change the name, as to not overwrite what is already there.

# Generally about 80% of images go into train and 20% into valid.
# Try to get about 50 images per class, this will help keep the accuracy high

import cv2
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
train = os.path.join(get_parent_dir(1), "Data", "Training_Images", "Train")
valid = os.path.join(get_parent_dir(1), "Data", "Training_Images", "Valid")

# initialising camera
vid = cv2.VideoCapture(0)

# initialising variables
number = 0

while True:
    # Load Images
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        number = number + 1
        # below is the output file,
        # valid is the directory, can change this to train if that's the aim.
        # make sure you change the name to that specific image dataset, e.g.: D6_One_WhiteBackground.jpg
        outfile = os.path.join(valid, 'picture_%s.jpg' % str(number))
        cv2.imwrite(outfile, frame)

    # closes file
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
