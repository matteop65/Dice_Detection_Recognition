# import show_camera
import pandas
import cv2
import sys
import dice_value
import dice_detect


def motion():
    # print("no roll...")
    # sys.stdout.write("no roll..." + "\n")
    # initiating static background to None -- so later it will be taken as the first frame
    static_back = None

    # Capturing video
    video = cv2.VideoCapture(0)
    rolled = 0
    detect = 1
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
            # print("hello")
            if cv2.contourArea(contour) > 1000:
                rolled = 1
                # print("rolling...")
                sys.stdout.write("rolling..." + "\n")
                detect = 0
                break
        break

    if rolled == 1:
        return rolled
    else:
        return None


def dice_detection():
    print("detecting...")
    # sys.stdout.write("detecting...\n")
    no_of_dice_3 = 0
    no_of_dice_3 = dice_detect.detect_dice()
    # print("Dice_detection: " + str(no_of_dice))
    return no_of_dice_3


def dice_value_return():
    print("evaluating...")
    # sys.stdout.write("evaluating...\n")
    dice_label = []
    no_of_dice = 0
    no_of_dice, dice_label = dice_value.evaluate_dice_value_()
    # dice_label = dice_value.r_d_labels()
    # no_of_dice = dice_value.r_d_value()
    # print("dice_value: " + str(no_of_dice))
    return no_of_dice, dice_label


if __name__ == "__main__":
    no_of_dice_1 = -2
    no_of_dice_2 = -3
    dice_labels = []
    print("starting...")
    # sys.stdout.write("starting..." + "\n")
    while True:
        roll_detected = motion()
        # print("Im here")
        if roll_detected == 1:
            while True:
                if no_of_dice_1 != no_of_dice_2:
                    # print("here")
                    no_of_dice_1 = dice_detection()
                    # print("no_dice_1: " + str(no_of_dice_1))
                    no_of_dice_2, dice_labels = dice_value_return()
                    # print("no_dice_2: " + str(no_of_dice_2))
                else:
                    break
            # print("\nNumber of dice:" + str(no_of_dice_2) + "\n")
            print("________________________________")
            for i in range(len(dice_labels)):
                print(dice_labels[i])
            print("________________________________")
            # print("Dice: " + str(dice_labels))
            dice_labels.clear()
            dice_labels = []
            no_of_dice_2 = -2
            no_of_dice_1 = -3
            roll_detected = 0
        else:
            continue
