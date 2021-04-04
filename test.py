import cv2


vs = cv2.VideoCapture(0)
firstFrame = None

while True:
    frame = vs.read()
    text = "No Dice"

    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_COLORCVT_MAX)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)