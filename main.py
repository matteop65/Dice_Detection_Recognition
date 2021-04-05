# Version 1.0
# Run this code in terminal (or in IDE).

import cv2
import dice_value
import dice_detect
import time


def motion():
    # This function detects whether the dice roll has started.
    # It does not detect when dice roll has finished. This saves resources as continuosly running the dice detection
    #       is very resourceful.
    # If the dice have not stopped yet, it will simple continue detecting and evaluating until they have.

    # initiating static background to None -- so later it will be taken as the first frame
    static_back = None

    # Capturing video
    video = cv2.VideoCapture(0)
    rolled = 0
    while True:
        # Reading frame(image) from video
        check, frame = video.read()

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # In first iteration we assign the value of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Difference between static background and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Finding contour of moving object
        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) > 800:
                rolled = 1
                print("rolling...")
                break
        break

    if rolled == 1:
        return rolled
    else:
        return None


def dice_detection():
    # calls dice detection function in dice_detect.py
    # returns the value of dice found.
    print("detecting...")
    no_of_dice_3 = dice_detect.detect_dice()
    return no_of_dice_3


def dice_value_return():
    print("evaluating...")
    # calls function in dice_value.py
    # returns number of dice found and dice labels for those dice
    no_of_dice, dice_label = dice_value.evaluate_dice_value_()
    return no_of_dice, dice_label


if __name__ == "__main__":
    no_of_dice_1 = -2
    no_of_dice_2 = -3
    dice_labels = []
    print("starting...")
    while True:
        roll_detected = motion()
        if roll_detected == 1:
            time.sleep(0.5)
            end = 0
            start = time.time()
            while True:
                end = time.time()

                # if algorithms, detect and evaluate, run for too long, restart process
                # This may happen is the detect algorithm is incorrect, which is possible due to the limited dataset
                if end - start >= 8:
                    print("________________________________")
                    print("Could not detect dice value")
                    print("Wave hand to recalculate dice value")
                    print("________________________________")
                    break

                # if the number of dice detected and number of dice_values detected are not equal, then re-run algorithm
                if no_of_dice_1 != no_of_dice_2:
                    no_of_dice_1 = dice_detection()
                    no_of_dice_2, dice_labels = dice_value_return()
                else:
                    break
            print("________________________________")
            print("\n")
            for i in range(len(dice_labels)):
                print(dice_labels[i])
            print("________________________________")
            dice_labels.clear()
            dice_labels = []
            no_of_dice_2 = -2
            no_of_dice_1 = -3
            roll_detected = 0
        else:
            continue
